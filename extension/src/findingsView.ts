import * as vscode from "vscode";
import { Analyzer, Finding } from "./diagnostics";

type Node = FileNode | FindingNode;

class FileNode {
  readonly kind = "file";
  constructor(
    readonly uri: vscode.Uri,
    readonly findings: Finding[],
  ) {}
}

class FindingNode {
  readonly kind = "finding";
  constructor(readonly finding: Finding) {}
}

/** Shows findings grouped by file, worst severity first. */
export class FindingsProvider implements vscode.TreeDataProvider<Node> {
  private readonly changed = new vscode.EventEmitter<Node | undefined>();
  readonly onDidChangeTreeData = this.changed.event;

  constructor(private readonly analyzer: Analyzer) {
    analyzer.onDidChangeFindings(() => this.changed.fire(undefined));
  }

  refresh(): void {
    this.changed.fire(undefined);
  }

  getTreeItem(node: Node): vscode.TreeItem {
    if (node.kind === "file") {
      const item = new vscode.TreeItem(
        vscode.workspace.asRelativePath(node.uri),
        vscode.TreeItemCollapsibleState.Expanded,
      );
      item.description = `${node.findings.length}`;
      item.resourceUri = node.uri;
      item.iconPath = vscode.ThemeIcon.File;
      return item;
    }
    const { match, uri, range } = node.finding;
    const item = new vscode.TreeItem(match.entity, vscode.TreeItemCollapsibleState.None);
    item.description = `line ${range.start.line + 1} · ${match.severity}${
      match.validated ? " · validated" : ""
    }`;
    item.iconPath = new vscode.ThemeIcon(
      match.validated ? "verified-filled" : "shield",
      new vscode.ThemeColor(
        match.severity === "critical" ? "editorError.foreground" : "editorWarning.foreground",
      ),
    );
    item.command = {
      command: "vscode.open",
      title: "Open",
      arguments: [uri, { selection: range }],
    };
    return item;
  }

  getChildren(node?: Node): Node[] {
    if (!node) {
      const files: FileNode[] = [];
      for (const [key, findings] of this.analyzer.allFindings()) {
        files.push(new FileNode(vscode.Uri.parse(key), findings));
      }
      files.sort((a, b) => b.findings.length - a.findings.length);
      return files;
    }
    if (node.kind === "file") {
      return node.findings.map((finding) => new FindingNode(finding));
    }
    return [];
  }
}
