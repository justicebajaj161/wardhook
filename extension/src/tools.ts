import * as path from "path";
import * as vscode from "vscode";
import { PiiMatch, Severity, getConfig } from "./config";
import { Analyzer } from "./diagnostics";
import { Sidecar } from "./sidecar";

interface TextSource {
  text?: string;
  filePath?: string;
  pack?: string;
}

/** Resolves a tool's `filePath`/`text` pair into a payload for the analyzer. */
async function resolveSource(
  input: TextSource,
): Promise<{ payload: Record<string, unknown>; label: string }> {
  if (input.filePath) {
    const uri = toUri(input.filePath);
    const document = await vscode.workspace.openTextDocument(uri);
    return {
      payload: { text: document.getText() },
      label: vscode.workspace.asRelativePath(uri),
    };
  }
  if (typeof input.text === "string" && input.text.length > 0) {
    return { payload: { text: input.text }, label: "the provided text" };
  }
  const active = vscode.window.activeTextEditor;
  if (active) {
    return {
      payload: { text: active.document.getText() },
      label: vscode.workspace.asRelativePath(active.document.uri),
    };
  }
  throw new Error("Give me a filePath or some text: no file is open to fall back to.");
}

function toUri(filePath: string): vscode.Uri {
  if (path.isAbsolute(filePath)) {
    return vscode.Uri.file(filePath);
  }
  const root = vscode.workspace.workspaceFolders?.[0];
  if (!root) {
    return vscode.Uri.file(filePath);
  }
  return vscode.Uri.joinPath(root.uri, filePath);
}

function result(text: string): vscode.LanguageModelToolResult {
  return new vscode.LanguageModelToolResult([new vscode.LanguageModelTextPart(text)]);
}

function packOf(input: TextSource): Record<string, unknown> {
  const config = getConfig();
  return {
    pack: input.pack ?? config.pack,
    customPackPath: input.pack ? undefined : config.customPackPath || undefined,
  };
}

/** Registers all seven tools and returns their disposables. */
export function registerTools(sidecar: Sidecar, analyzer: Analyzer): vscode.Disposable[] {
  const lm = vscode.lm as typeof vscode.lm | undefined;
  if (!lm?.registerTool) {
    // An older editor, or a fork without the tools API. Everything else works.
    return [];
  }

  const scanPii = lm.registerTool<TextSource>("wardhook_scan_pii", {
    async invoke(options, token) {
      const { payload, label } = await resolveSource(options.input);
      const res = await sidecar.request<{ matches: PiiMatch[]; length: number }>(
        "scan",
        { ...payload, ...packOf(options.input) },
        token,
      );
      if (res.matches.length === 0) {
        return result(`No personal data detected in ${label}.`);
      }
      const lines = res.matches.map(
        (m) =>
          `- ${m.entity} at line ${m.startLine + 1}, column ${m.startChar + 1} ` +
          `(severity ${m.severity}${m.validated ? ", checksum-validated" : ""})`,
      );
      return result(
        `${res.matches.length} finding(s) in ${label}:\n${lines.join("\n")}\n\n` +
          "The matched values are deliberately not returned by the detector.",
      );
    },
  });

  const redact = lm.registerTool<TextSource>("wardhook_redact", {
    async invoke(options, token) {
      const { payload, label } = await resolveSource(options.input);
      const res = await sidecar.request<{
        text: string;
        counts: Record<string, number>;
        matchCount: number;
      }>("redact", { ...payload, ...packOf(options.input) }, token);
      if (res.matchCount === 0) {
        return result(`Nothing to redact in ${label}.`);
      }
      const counts = Object.entries(res.counts)
        .map(([entity, n]) => `${entity} x${n}`)
        .join(", ");
      return result(`Redacted ${res.matchCount} item(s) in ${label} (${counts}):\n\n${res.text}`);
    },
  });

  const scanWorkspace = lm.registerTool<{ include?: string; maxFiles?: number }>(
    "wardhook_scan_workspace",
    {
      async invoke(options, token) {
        const config = getConfig();
        const include =
          options.input.include ?? "**/*.{py,md,yaml,yml,json,jsonl,txt,ts,js,env,ini,toml,sql,sh}";
        const uris = await vscode.workspace.findFiles(
          include,
          `{${[...config.excludeGlobs, "**/node_modules/**", "**/.git/**"].join(",")}}`,
          options.input.maxFiles ?? 500,
        );
        const byEntity = new Map<string, number>();
        const bySeverity = new Map<string, number>();
        const perFile: { file: string; count: number }[] = [];
        let total = 0;
        for (const uri of uris) {
          if (token.isCancellationRequested) {
            break;
          }
          const document = await vscode.workspace.openTextDocument(uri);
          const res = await sidecar.request<{ matches: PiiMatch[] }>(
            "scan",
            { text: document.getText(), pack: config.pack },
            token,
          );
          if (res.matches.length === 0) {
            continue;
          }
          total += res.matches.length;
          perFile.push({ file: vscode.workspace.asRelativePath(uri), count: res.matches.length });
          for (const m of res.matches) {
            byEntity.set(m.entity, (byEntity.get(m.entity) ?? 0) + 1);
            bySeverity.set(m.severity, (bySeverity.get(m.severity) ?? 0) + 1);
          }
        }
        if (total === 0) {
          return result(`Scanned ${uris.length} file(s). No personal data detected.`);
        }
        const top = perFile
          .sort((a, b) => b.count - a.count)
          .slice(0, 10)
          .map((f) => `- ${f.file}: ${f.count}`);
        const entities = [...byEntity.entries()]
          .sort((a, b) => b[1] - a[1])
          .map(([e, n]) => `${e} x${n}`)
          .join(", ");
        const severities = [...bySeverity.entries()].map(([s, n]) => `${s} x${n}`).join(", ");
        return result(
          `${total} finding(s) across ${perFile.length} of ${uris.length} scanned file(s).\n` +
            `By entity: ${entities}\nBy severity: ${severities}\n\nWorst files:\n${top.join("\n")}`,
        );
      },
    },
  );

  const scoreInjection = lm.registerTool<TextSource & { threshold?: number }>(
    "wardhook_score_injection",
    {
      async invoke(options, token) {
        const { payload, label } = await resolveSource(options.input);
        const threshold = options.input.threshold ?? getConfig().injectionThreshold;
        const res = await sidecar.request<{
          score: number;
          threshold: number;
          categories: string[];
          severity: Severity;
          signals: { category: string; weight: number; patterns_hit: number }[];
        }>("injection", { ...payload, threshold }, token);
        if (res.categories.length === 0) {
          return result(
            `No prompt-injection signals in ${label}. Score 0.00 against a threshold of ${res.threshold}.`,
          );
        }
        const signals = res.signals
          .map((s) => `- ${s.category} (weight ${s.weight}, ${s.patterns_hit} pattern hit(s))`)
          .join("\n");
        return result(
          `Prompt-injection score for ${label}: ${res.score.toFixed(2)} ` +
            `(threshold ${res.threshold}, severity ${res.severity}).\n` +
            `Signals that fired:\n${signals}\n\n` +
            "This is a whole-text score with no character offsets, so there is no span to highlight.",
        );
      },
    },
  );

  const listPacks = lm.registerTool<Record<string, never>>("wardhook_list_packs", {
    async invoke(_options, token) {
      const res = await sidecar.request<{
        packs: { name: string; entities: string[]; ruleCount: number }[];
      }>("packs", {}, token);
      const active = getConfig().pack;
      const lines = res.packs.map(
        (p) =>
          `- ${p.name}${p.name === active ? " (active)" : ""}: ${p.ruleCount} rules — ` +
          p.entities.join(", "),
      );
      return result(`Available entity packs:\n${lines.join("\n")}`);
    },
  });

  const explainEntity = lm.registerTool<{ entity: string; pack?: string }>(
    "wardhook_explain_entity",
    {
      async invoke(options, token) {
        const res = await sidecar.request<any>(
          "explain",
          { entity: options.input.entity, pack: options.input.pack ?? getConfig().pack },
          token,
        );
        if (!res.found) {
          return result(
            `No entity called ${options.input.entity} in the ${res.pack} pack. ` +
              `It knows: ${res.known.join(", ")}.`,
          );
        }
        const validator = res.validator
          ? `confirmed by a ${res.validator} checksum`
          : "pattern-only, with no checksum to confirm it";
        const context = res.contextWords?.length
          ? ` Requires one of these words nearby: ${res.contextWords.join(", ")}.`
          : "";
        return result(
          `${res.entity} (${res.pack} pack): ${res.description}\n` +
            `Severity ${res.severity}, ${validator}. Redacted as ${res.replacement}.${context}`,
        );
      },
    },
  );

  const checkPolicy = lm.registerTool<{
    role: string;
    tool: string;
    policy: Record<string, string[]>;
  }>("wardhook_check_tool_policy", {
    async invoke(options, token) {
      const res = await sidecar.request<{
        allowed: boolean;
        reason: string | null;
        grantedPatterns: string[];
      }>(
        "policy",
        { role: options.input.role, tool: options.input.tool, policy: options.input.policy },
        token,
      );
      const granted = res.grantedPatterns.length
        ? res.grantedPatterns.join(", ")
        : "nothing at all";
      return result(
        res.allowed
          ? `Allowed: role "${options.input.role}" may call "${options.input.tool}". ` +
              `That role grants: ${granted}.`
          : `Denied: ${res.reason ?? "no grant matched"}. That role grants: ${granted}. ` +
              "The policy is deny-by-default.",
      );
    },
  });

  void analyzer;
  return [scanPii, redact, scanWorkspace, scoreInjection, listPacks, explainEntity, checkPolicy];
}
