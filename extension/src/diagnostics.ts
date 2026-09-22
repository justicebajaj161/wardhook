import * as vscode from "vscode";
import {
  PiiMatch,
  getConfig,
  isExcluded,
  shouldReport,
  toDiagnosticSeverity,
} from "./config";
import { Sidecar } from "./sidecar";

export const SOURCE = "wardhook";

/** Languages worth scanning. Binary and generated formats are pointless. */
const SCANNABLE = new Set([
  "python",
  "markdown",
  "yaml",
  "json",
  "jsonc",
  "jsonl",
  "plaintext",
  "javascript",
  "typescript",
  "typescriptreact",
  "javascriptreact",
  "toml",
  "ini",
  "dotenv",
  "shellscript",
  "sql",
  "log",
]);

/** A diagnostic plus the match that produced it, kept for quick fixes. */
export interface Finding {
  uri: vscode.Uri;
  match: PiiMatch;
  range: vscode.Range;
}

/**
 * Scans documents and publishes findings.
 *
 * Owns the debounce, the reporting filter, the diagnostic collection, the
 * status bar item, and the findings the tree view renders.
 */
export class Analyzer implements vscode.Disposable {
  private readonly collection: vscode.DiagnosticCollection;
  private readonly status: vscode.StatusBarItem;
  private readonly timers = new Map<string, NodeJS.Timeout>();
  private readonly cancels = new Map<string, vscode.CancellationTokenSource>();
  private readonly findings = new Map<string, Finding[]>();
  private readonly changed = new vscode.EventEmitter<void>();
  private warned = false;

  /** Fires whenever the findings change, so views can refresh. */
  readonly onDidChangeFindings = this.changed.event;

  constructor(
    private readonly sidecar: Sidecar,
    private readonly log: vscode.OutputChannel,
    private readonly onFailure?: (error: unknown) => void,
  ) {
    this.collection = vscode.languages.createDiagnosticCollection(SOURCE);
    this.status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    this.status.command = "wardhook.findings.focus";
    this.refreshStatus();
  }

  /** True when this document is worth sending to the analyzer. */
  isScannable(document: vscode.TextDocument): boolean {
    if (document.uri.scheme !== "file" || document.isUntitled) {
      return SCANNABLE.has(document.languageId) && document.uri.scheme === "untitled";
    }
    if (!SCANNABLE.has(document.languageId)) {
      return false;
    }
    const config = getConfig(document.uri);
    if (isExcluded(document.uri, config)) {
      return false;
    }
    return document.getText().length <= config.maxFileSizeKb * 1024;
  }

  /** Queues a scan after the configured idle period. */
  schedule(document: vscode.TextDocument): void {
    const key = document.uri.toString();
    const config = getConfig(document.uri);
    const existing = this.timers.get(key);
    if (existing) {
      clearTimeout(existing);
    }
    this.timers.set(
      key,
      setTimeout(() => {
        this.timers.delete(key);
        void this.scan(document);
      }, Math.max(0, config.debounceMs)),
    );
  }

  /** Scans one document and republishes its diagnostics. */
  async scan(document: vscode.TextDocument): Promise<Finding[]> {
    const key = document.uri.toString();
    if (!this.isScannable(document)) {
      this.clear(document.uri);
      return [];
    }
    // Supersede any scan still in flight for this document.
    this.cancels.get(key)?.cancel();
    const source = new vscode.CancellationTokenSource();
    this.cancels.set(key, source);

    const config = getConfig(document.uri);
    try {
      const result = await this.sidecar.request<{ matches: PiiMatch[] }>(
        "scan",
        {
          text: document.getText(),
          pack: config.pack,
          customPackPath: config.customPackPath || undefined,
        },
        source.token,
      );
      const findings = result.matches
        .filter((match) => shouldReport(match, config))
        .map((match) => ({
          uri: document.uri,
          match,
          range: new vscode.Range(
            match.startLine,
            match.startChar,
            match.endLine,
            match.endChar,
          ),
        }));
      this.publish(document.uri, findings);
      return findings;
    } catch (error) {
      if (error instanceof vscode.CancellationError) {
        return [];
      }
      this.log.appendLine(`[scan] ${document.uri.fsPath}: ${String(error)}`);
      // Missing squiggles look like "nothing found", which is the most
      // dangerous way for this extension to fail. Say so once.
      this.reportOnce(error);
      return [];
    } finally {
      if (this.cancels.get(key) === source) {
        this.cancels.delete(key);
      }
      source.dispose();
    }
  }

  /** Scans every matching file in the workspace. */
  async scanWorkspace(
    progress?: vscode.Progress<{ message?: string; increment?: number }>,
    token?: vscode.CancellationToken,
  ): Promise<Finding[]> {
    const config = getConfig();
    const uris = await vscode.workspace.findFiles(
      "**/*.{py,md,yaml,yml,json,jsonl,txt,ts,js,env,ini,toml,sql,sh,log}",
      `{${[...config.excludeGlobs, "**/node_modules/**", "**/.git/**"].join(",")}}`,
      2000,
    );
    const all: Finding[] = [];
    let done = 0;
    for (const uri of uris) {
      if (token?.isCancellationRequested) {
        break;
      }
      done += 1;
      progress?.report({
        message: `${done}/${uris.length}  ${vscode.workspace.asRelativePath(uri)}`,
        increment: 100 / uris.length,
      });
      try {
        const document = await vscode.workspace.openTextDocument(uri);
        all.push(...(await this.scan(document)));
      } catch (error) {
        this.log.appendLine(`[scanWorkspace] skipped ${uri.fsPath}: ${String(error)}`);
      }
    }
    return all;
  }

  private publish(uri: vscode.Uri, findings: Finding[]): void {
    const diagnostics = findings.map((finding) => {
      const { match } = finding;
      const confidence = match.validated ? "checksum-validated" : "pattern match";
      const diagnostic = new vscode.Diagnostic(
        finding.range,
        `${match.entity} detected (${match.severity}, ${confidence}).`,
        toDiagnosticSeverity(match.severity),
      );
      diagnostic.source = SOURCE;
      diagnostic.code = match.entity;
      return diagnostic;
    });
    this.collection.set(uri, diagnostics);
    if (findings.length > 0) {
      this.findings.set(uri.toString(), findings);
    } else {
      this.findings.delete(uri.toString());
    }
    this.refreshStatus();
    this.changed.fire();
  }

  /** Drops everything recorded for a file, e.g. when it is closed. */
  clear(uri: vscode.Uri): void {
    this.collection.delete(uri);
    this.findings.delete(uri.toString());
    this.refreshStatus();
    this.changed.fire();
  }

  /** Clears the failure latch so scanning can report again after a fix. */
  resetFailure(): void {
    this.warned = false;
    this.refreshStatus();
  }

  clearAll(): void {
    this.collection.clear();
    this.findings.clear();
    this.refreshStatus();
    this.changed.fire();
  }

  /** Findings for one file, used by the quick-fix provider. */
  findingsFor(uri: vscode.Uri): Finding[] {
    return this.findings.get(uri.toString()) ?? [];
  }

  /** Every file with findings, newest scan order, for the tree view. */
  allFindings(): Map<string, Finding[]> {
    return this.findings;
  }

  /** Surfaces the first scan failure, so a broken analyzer is not silent. */
  private reportOnce(error: unknown): void {
    if (this.warned) {
      return;
    }
    this.warned = true;
    if (this.onFailure) {
      // The host knows how to offer a fix for a missing dependency; a bare
      // error notification would just tell the user to go and read a log.
      this.status.text = "$(shield) Wardhook: unavailable";
      this.status.tooltip = String(error);
      this.status.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
      this.onFailure(error);
      return;
    }
    this.status.text = "$(shield) Wardhook: unavailable";
    this.status.tooltip = String(error);
    this.status.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
    void vscode.window
      .showErrorMessage(`Wardhook could not scan: ${errorText(error)}`, "Show Log")
      .then((choice) => {
        if (choice === "Show Log") {
          this.log.show();
        }
      });
  }

  private refreshStatus(): void {
    if (this.warned) {
      return;
    }
    let total = 0;
    for (const list of this.findings.values()) {
      total += list.length;
    }
    if (total === 0) {
      this.status.text = "$(shield) Wardhook";
      this.status.tooltip = "No personal data found in the scanned files.";
      this.status.backgroundColor = undefined;
    } else {
      this.status.text = `$(shield) Wardhook: ${total}`;
      this.status.tooltip = `${total} finding${total === 1 ? "" : "s"}. Click to review.`;
      this.status.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
    }
    this.status.show();
  }

  dispose(): void {
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    for (const source of this.cancels.values()) {
      source.cancel();
      source.dispose();
    }
    this.timers.clear();
    this.cancels.clear();
    this.collection.dispose();
    this.status.dispose();
    this.changed.dispose();
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
