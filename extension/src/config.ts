import * as vscode from "vscode";

export type Severity = "low" | "medium" | "high" | "critical";

export const SEVERITY_ORDER: Record<Severity, number> = {
  low: 0,
  medium: 1,
  high: 2,
  critical: 3,
};

export interface WardhookConfig {
  pack: string;
  customPackPath: string;
  minSeverity: Severity;
  validatedOnly: boolean;
  scanOnType: boolean;
  debounceMs: number;
  excludeGlobs: string[];
  excludeEntities: string[];
  maxFileSizeKb: number;
  injectionThreshold: number;
  pythonPath: string;
  policyPackRepo: string;
  policyPackPath: string;
}

/** Reads the current settings. Called per scan so changes apply immediately. */
export function getConfig(scope?: vscode.Uri): WardhookConfig {
  const c = vscode.workspace.getConfiguration("wardhook", scope ?? null);
  return {
    pack: c.get<string>("pack", "default"),
    customPackPath: c.get<string>("customPackPath", ""),
    minSeverity: c.get<Severity>("minSeverity", "high"),
    validatedOnly: c.get<boolean>("validatedOnly", false),
    scanOnType: c.get<boolean>("scanOnType", true),
    debounceMs: c.get<number>("debounceMs", 150),
    excludeGlobs: c.get<string[]>("excludeGlobs", []),
    excludeEntities: c.get<string[]>("excludeEntities", []),
    maxFileSizeKb: c.get<number>("maxFileSizeKb", 1024),
    injectionThreshold: c.get<number>("injectionThreshold", 0.5),
    pythonPath: c.get<string>("pythonPath", ""),
    policyPackRepo: c.get<string>("policyPackRepo", ""),
    policyPackPath: c.get<string>("policyPackPath", "wardhook-pack.yaml"),
  };
}

/** One detected entity, as the analyzer reports it. Never carries the value. */
export interface PiiMatch {
  entity: string;
  start: number;
  end: number;
  length: number;
  severity: Severity;
  validated: boolean;
  replacement: string;
  startLine: number;
  startChar: number;
  endLine: number;
  endChar: number;
}

/**
 * Decides whether a match is worth showing.
 *
 * Scanning this repo raw yields 89 matches, 79 of which are docstring examples
 * and default config values. Keeping checksum-validated matches plus anything
 * at or above `minSeverity` cuts that to 8 without losing a real leak.
 */
export function shouldReport(match: PiiMatch, config: WardhookConfig): boolean {
  if (config.excludeEntities.includes(match.entity)) {
    return false;
  }
  if (config.validatedOnly) {
    return match.validated;
  }
  if (match.validated) {
    return true;
  }
  return SEVERITY_ORDER[match.severity] >= SEVERITY_ORDER[config.minSeverity];
}

/** Maps a Wardhook severity onto the editor's four diagnostic levels. */
export function toDiagnosticSeverity(severity: Severity): vscode.DiagnosticSeverity {
  switch (severity) {
    case "critical":
      return vscode.DiagnosticSeverity.Error;
    case "high":
      return vscode.DiagnosticSeverity.Warning;
    case "medium":
      return vscode.DiagnosticSeverity.Information;
    default:
      return vscode.DiagnosticSeverity.Hint;
  }
}

/** True when a path matches any configured exclude glob. */
export function isExcluded(uri: vscode.Uri, config: WardhookConfig): boolean {
  const relative = vscode.workspace.asRelativePath(uri, false);
  return config.excludeGlobs.some((glob) => matchGlob(relative, glob) || matchGlob(uri.fsPath, glob));
}

/** Minimal glob matcher covering the `**`, `*` and `?` forms used in settings. */
export function matchGlob(value: string, glob: string): boolean {
  const normalized = value.replace(/\\/g, "/");
  let pattern = "";
  let i = 0;
  while (i < glob.length) {
    const char = glob[i];
    if (char === "*") {
      if (glob[i + 1] === "*") {
        // `**/` spans any number of directories, including none at all.
        if (glob[i + 2] === "/") {
          pattern += "(?:.*/)?";
          i += 3;
        } else {
          pattern += ".*";
          i += 2;
        }
      } else {
        pattern += "[^/]*";
        i += 1;
      }
    } else if (char === "?") {
      pattern += "[^/]";
      i += 1;
    } else {
      pattern += char.replace(/[.+^${}()|[\]\\]/g, "\\$&");
      i += 1;
    }
  }
  return new RegExp(`^${pattern}$`).test(normalized);
}
