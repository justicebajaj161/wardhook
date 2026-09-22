import * as vscode from "vscode";
import { Analyzer, SOURCE } from "./diagnostics";

/**
 * Offers redactions for the findings under the cursor.
 *
 * The replacement text comes from the analyzer rather than being rebuilt here,
 * so the placeholder a quick fix inserts is exactly the one the library would
 * have written itself.
 */
export class RedactionActions implements vscode.CodeActionProvider {
  static readonly providedCodeActionKinds = [vscode.CodeActionKind.QuickFix];

  constructor(private readonly analyzer: Analyzer) {}

  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range | vscode.Selection,
    context: vscode.CodeActionContext,
  ): vscode.CodeAction[] {
    const findings = this.analyzer.findingsFor(document.uri);
    if (findings.length === 0) {
      return [];
    }
    const actions: vscode.CodeAction[] = [];
    const here = findings.filter((finding) => finding.range.intersection(range) !== undefined);

    for (const finding of here) {
      const fix = new vscode.CodeAction(
        `Redact this ${finding.match.entity}`,
        vscode.CodeActionKind.QuickFix,
      );
      fix.edit = new vscode.WorkspaceEdit();
      fix.edit.replace(document.uri, finding.range, finding.match.replacement);
      fix.diagnostics = context.diagnostics.filter(
        (d) => d.source === SOURCE && d.range.isEqual(finding.range),
      );
      fix.isPreferred = true;
      actions.push(fix);

      const ignore = new vscode.CodeAction(
        `Stop reporting ${finding.match.entity} in this workspace`,
        vscode.CodeActionKind.QuickFix,
      );
      ignore.command = {
        command: "wardhook.ignoreEntity",
        title: "Ignore entity",
        arguments: [finding.match.entity],
      };
      actions.push(ignore);
    }

    if (findings.length > 1) {
      const all = new vscode.CodeAction(
        `Redact all ${findings.length} findings in this file`,
        vscode.CodeActionKind.QuickFix,
      );
      all.edit = new vscode.WorkspaceEdit();
      // Applied last-first so each edit's offsets stay valid as text shrinks.
      const ordered = [...findings].sort((a, b) => b.match.start - a.match.start);
      for (const finding of ordered) {
        all.edit.replace(document.uri, finding.range, finding.match.replacement);
      }
      actions.push(all);
    }
    return actions;
  }
}
