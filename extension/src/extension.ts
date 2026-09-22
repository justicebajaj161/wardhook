import * as path from "path";
import * as vscode from "vscode";
import { getSession, syncPolicyPack } from "./auth";
import { PiiMatch, getConfig } from "./config";
import { RedactionActions } from "./codeActions";
import { Analyzer } from "./diagnostics";
import { FindingsProvider } from "./findingsView";
import { registerParticipant } from "./participant";
import {
  MissingDependencyError,
  Sidecar,
  installGuardrails,
} from "./sidecar";
import { registerTools } from "./tools";

export function activate(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel("Wardhook");
  const scriptPath = path.join(context.extensionPath, "python", "wardhook_sidecar.py");
  const sidecar = new Sidecar(scriptPath, log);
  const analyzer = new Analyzer(sidecar, log, (error) => void onAnalyzerFailure(error));
  const findings = new FindingsProvider(analyzer);

  context.subscriptions.push(log, sidecar, analyzer);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("wardhook.findings", findings),
  );
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      { scheme: "file" },
      new RedactionActions(analyzer),
      { providedCodeActionKinds: RedactionActions.providedCodeActionKinds },
    ),
  );

  // Chat surfaces are optional: absent in a fork or without Copilot installed.
  const participant = registerParticipant(context, sidecar, analyzer);
  if (participant) {
    context.subscriptions.push(participant);
  }
  context.subscriptions.push(...registerTools(sidecar, analyzer));

  // --- document lifecycle -------------------------------------------------
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((document) => analyzer.schedule(document)),
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (getConfig(event.document.uri).scanOnType) {
        analyzer.schedule(event.document);
      }
    }),
    vscode.workspace.onDidSaveTextDocument((document) => void analyzer.scan(document)),
    vscode.workspace.onDidCloseTextDocument((document) => analyzer.clear(document.uri)),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (!event.affectsConfiguration("wardhook")) {
        return;
      }
      if (event.affectsConfiguration("wardhook.pythonPath")) {
        sidecar.restart();
      }
      analyzer.clearAll();
      for (const document of vscode.workspace.textDocuments) {
        analyzer.schedule(document);
      }
    }),
  );

  // --- commands -----------------------------------------------------------
  const command = (name: string, run: (...args: any[]) => unknown) =>
    context.subscriptions.push(
      vscode.commands.registerCommand(name, async (...args: any[]) => {
        try {
          return await run(...args);
        } catch (error) {
          log.appendLine(`[${name}] ${String(error)}`);
          void vscode.window.showErrorMessage(`Wardhook: ${errorText(error)}`);
          return undefined;
        }
      }),
    );

  command("wardhook.scanFile", async () => {
    const editor = requireEditor();
    const found = await analyzer.scan(editor.document);
    void vscode.window.showInformationMessage(
      found.length === 0
        ? "Wardhook: no personal data found."
        : `Wardhook: ${found.length} finding(s).`,
    );
  });

  command("wardhook.scanWorkspace", async () => {
    await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: "Wardhook: scanning", cancellable: true },
      async (progress, token) => {
        const found = await analyzer.scanWorkspace(progress, token);
        void vscode.window.showInformationMessage(
          `Wardhook: ${found.length} finding(s) across the workspace.`,
        );
      },
    );
  });

  command("wardhook.redactSelection", async () => {
    const editor = requireEditor();
    const range = editor.selection.isEmpty
      ? new vscode.Range(
          editor.document.positionAt(0),
          editor.document.positionAt(editor.document.getText().length),
        )
      : editor.selection;
    const res = await sidecar.request<{ text: string; matchCount: number }>("redact", {
      text: editor.document.getText(range),
      pack: getConfig(editor.document.uri).pack,
    });
    if (res.matchCount === 0) {
      void vscode.window.showInformationMessage("Wardhook: nothing to redact.");
      return;
    }
    await editor.edit((builder) => builder.replace(range, res.text));
    void vscode.window.showInformationMessage(`Wardhook: redacted ${res.matchCount} item(s).`);
  });

  command("wardhook.redactFile", async () => {
    const editor = requireEditor();
    const whole = new vscode.Range(
      editor.document.positionAt(0),
      editor.document.positionAt(editor.document.getText().length),
    );
    const res = await sidecar.request<{ text: string; matchCount: number }>("redact", {
      text: editor.document.getText(),
      pack: getConfig(editor.document.uri).pack,
    });
    if (res.matchCount === 0) {
      void vscode.window.showInformationMessage("Wardhook: nothing to redact.");
      return;
    }
    await editor.edit((builder) => builder.replace(whole, res.text));
  });

  command("wardhook.scoreInjection", async () => {
    const editor = requireEditor();
    const res = await sidecar.request<{
      score: number;
      threshold: number;
      categories: string[];
      severity: string;
    }>("injection", {
      text: editor.document.getText(),
      threshold: getConfig(editor.document.uri).injectionThreshold,
    });
    void vscode.window.showInformationMessage(
      res.categories.length === 0
        ? "Wardhook: no prompt-injection signals."
        : `Wardhook: injection score ${res.score.toFixed(2)} (${res.severity}) — ` +
            res.categories.join(", "),
    );
  });

  command("wardhook.selectPack", async () => {
    const res = await sidecar.request<{ packs: { name: string; ruleCount: number }[] }>("packs");
    const picked = await vscode.window.showQuickPick(
      res.packs.map((pack) => ({ label: pack.name, description: `${pack.ruleCount} rules` })),
      { title: "Wardhook entity pack" },
    );
    if (picked) {
      await vscode.workspace
        .getConfiguration("wardhook")
        .update("pack", picked.label, vscode.ConfigurationTarget.Workspace);
    }
  });

  command("wardhook.ignoreEntity", async (entity: string) => {
    const config = vscode.workspace.getConfiguration("wardhook");
    const current = config.get<string[]>("excludeEntities", []);
    if (!current.includes(entity)) {
      await config.update(
        "excludeEntities",
        [...current, entity],
        vscode.ConfigurationTarget.Workspace,
      );
    }
  });

  command("wardhook.restartSidecar", async () => {
    sidecar.restart();
    analyzer.clearAll();
    for (const document of vscode.workspace.textDocuments) {
      analyzer.schedule(document);
    }
    void vscode.window.showInformationMessage("Wardhook: analyzer restarted.");
  });

  command("wardhook.showOutput", () => log.show());
  command("wardhook.refreshFindings", () => findings.refresh());

  command("wardhook.signInGitHub", async () => {
    const session = await getSession(true);
    if (session) {
      void vscode.window.showInformationMessage(`Wardhook: signed in as ${session.account.label}.`);
    }
  });

  command("wardhook.syncPolicyPack", async () => {
    const target = await syncPolicyPack(context.globalStorageUri);
    if (target) {
      void vscode.window.showInformationMessage(`Wardhook: entity pack synced to ${target.fsPath}.`);
    }
  });

  /**
   * Turns a missing dependency into one click rather than a log to read.
   *
   * The library is a single pure-Python package, so installing it is quick and
   * safe to offer -- and without it the extension can do nothing at all.
   */
  async function onAnalyzerFailure(error: unknown): Promise<void> {
    if (!(error instanceof MissingDependencyError)) {
      const choice = await vscode.window.showErrorMessage(
        `Wardhook could not scan: ${errorText(error)}`,
        "Show Log",
      );
      if (choice === "Show Log") {
        log.show();
      }
      return;
    }
    if (!error.installTarget) {
      const choice = await vscode.window.showErrorMessage(
        "Wardhook needs Python 3.10 or newer, and could not find one. " +
          "Install Python, or set wardhook.pythonPath.",
        "Open Settings",
        "Show Log",
      );
      if (choice === "Open Settings") {
        void vscode.commands.executeCommand("workbench.action.openSettings", "wardhook.pythonPath");
      } else if (choice === "Show Log") {
        log.show();
      }
      return;
    }
    const choice = await vscode.window.showWarningMessage(
      `Wardhook needs the wardhook-guardrails package. Install it into ${error.installTarget}?`,
      "Install",
      "Choose Interpreter",
      "Show Log",
    );
    if (choice === "Show Log") {
      log.show();
      return;
    }
    if (choice === "Choose Interpreter") {
      void vscode.commands.executeCommand("python.setInterpreter");
      return;
    }
    if (choice !== "Install") {
      return;
    }
    const installed = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Wardhook: installing wardhook-guardrails",
      },
      () => installGuardrails(error.installTarget as string, log),
    );
    if (!installed.ok) {
      const retry = await vscode.window.showErrorMessage(
        "Installing wardhook-guardrails failed.",
        "Show Log",
      );
      if (retry === "Show Log") {
        log.show();
      }
      return;
    }
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("pythonPath", error.installTarget, vscode.ConfigurationTarget.Global);
    sidecar.restart();
    analyzer.resetFailure();
    for (const document of vscode.workspace.textDocuments) {
      analyzer.schedule(document);
    }
    void vscode.window.showInformationMessage("Wardhook is ready.");
  }

  // Scan whatever is already open, so the first impression is not an empty panel.
  for (const document of vscode.workspace.textDocuments) {
    analyzer.schedule(document);
  }
  log.appendLine("[wardhook] extension activated");
}

export function deactivate(): void {
  // Everything is disposed through context.subscriptions.
}

function requireEditor(): vscode.TextEditor {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    throw new Error("open a file first.");
  }
  return editor;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export type { PiiMatch };
