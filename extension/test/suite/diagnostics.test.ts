import * as assert from "assert";
import * as path from "path";
import * as vscode from "vscode";
import { Analyzer } from "../../src/diagnostics";
import { Sidecar } from "../../src/sidecar";

const ROOT = path.join(__dirname, "..", "..", "..");
const SCRIPT = path.join(ROOT, "python", "wardhook_sidecar.py");
const FIXTURE = path.join(ROOT, "demo", "leaky_agent.py");

suite("diagnostics on the demo fixture", () => {
  let log: vscode.OutputChannel;
  let sidecar: Sidecar;
  let analyzer: Analyzer;
  let document: vscode.TextDocument;

  suiteSetup(async () => {
    log = vscode.window.createOutputChannel("Wardhook Test Diagnostics");
    sidecar = new Sidecar(SCRIPT, log);
    analyzer = new Analyzer(sidecar, log);
    document = await vscode.workspace.openTextDocument(vscode.Uri.file(FIXTURE));
    // The fixture lives outside the excluded test directories, but pin the
    // setting so a change to the defaults cannot silently empty this suite.
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("excludeGlobs", [], vscode.ConfigurationTarget.Global);
  });

  suiteTeardown(async () => {
    analyzer.dispose();
    sidecar.dispose();
    log.dispose();
    const config = vscode.workspace.getConfiguration("wardhook");
    await config.update("excludeGlobs", undefined, vscode.ConfigurationTarget.Global);
    await config.update("minSeverity", undefined, vscode.ConfigurationTarget.Global);
  });

  test("reports exactly the three real leaks at default settings", async () => {
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("minSeverity", "high", vscode.ConfigurationTarget.Global);
    const findings = await analyzer.scan(document);
    const entities = findings.map((f) => f.match.entity).sort();
    assert.deepStrictEqual(entities, ["AWS_ACCESS_KEY", "CREDIT_CARD", "US_SSN"]);
  });

  test("suppresses the medium-severity email and phone", async () => {
    const findings = await analyzer.scan(document);
    const entities = findings.map((f) => f.match.entity);
    assert.ok(!entities.includes("EMAIL"));
    assert.ok(!entities.includes("PHONE"));
  });

  test("lowering minSeverity surfaces all five", async () => {
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("minSeverity", "medium", vscode.ConfigurationTarget.Global);
    const findings = await analyzer.scan(document);
    assert.strictEqual(findings.length, 5, "expected every detected entity to be reported");
  });

  test("a range selects the text the detector matched", async () => {
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("minSeverity", "high", vscode.ConfigurationTarget.Global);
    const findings = await analyzer.scan(document);
    const ssn = findings.find((f) => f.match.entity === "US_SSN");
    assert.ok(ssn);
    assert.strictEqual(document.getText(ssn.range), "796-30-8562");
  });

  test("published diagnostics carry the entity as their code", async () => {
    await analyzer.scan(document);
    const diagnostics = vscode.languages
      .getDiagnostics(document.uri)
      .filter((d) => d.source === "wardhook");
    assert.ok(diagnostics.length >= 3, `expected at least 3, got ${diagnostics.length}`);
    assert.ok(diagnostics.every((d) => typeof d.code === "string"));
    // The activated extension publishes its own collection for this file too,
    // so compare the distinct entity codes rather than a raw count.
    const codes = [...new Set(diagnostics.map((d) => String(d.code)))].sort();
    assert.deepStrictEqual(codes, ["AWS_ACCESS_KEY", "CREDIT_CARD", "US_SSN"]);
  });

  test("the quick fix replaces a span with the library's placeholder", async () => {
    const findings = await analyzer.scan(document);
    const card = findings.find((f) => f.match.entity === "CREDIT_CARD");
    assert.ok(card);
    assert.strictEqual(card.match.replacement, "[CREDIT_CARD]");
    assert.strictEqual(document.getText(card.range), "4111 1111 1111 1111");
  });
});
