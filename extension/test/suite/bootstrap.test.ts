import * as assert from "assert";
import { execFile } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { promisify } from "util";
import * as vscode from "vscode";
import {
  MissingDependencyError,
  findInstallTarget,
  forgetInterpreter,
  resolvePython,
} from "../../src/sidecar";

const run = promisify(execFile);

/**
 * A Python that exists but has nothing installed — the state of a colleague's
 * laptop, and the one case where this extension is useless unless it helps.
 */
async function makeBareInterpreter(): Promise<string> {
  const dir = path.join(os.tmpdir(), `wardhook-bare-${Date.now()}`);
  const host = await resolvePython();
  await run(host, ["-m", "venv", "--without-pip", dir]);
  const python = path.join(dir, "bin", "python");
  assert.ok(fs.existsSync(python), "failed to build a bare interpreter");
  return python;
}

suite("first run on a machine without the library", () => {
  let bare: string;
  const config = () => vscode.workspace.getConfiguration("wardhook");

  suiteSetup(async function () {
    this.timeout(120000);
    bare = await makeBareInterpreter();
  });

  teardown(async () => {
    await config().update("pythonPath", undefined, vscode.ConfigurationTarget.Global);
    forgetInterpreter();
  });

  test("the bare interpreter really cannot import the library", async () => {
    await assert.rejects(() => run(bare, ["-c", "import wardhook.guardrails"]));
  });

  test("discovery reports a missing dependency rather than a vague failure", async function () {
    this.timeout(60000);
    await config().update("pythonPath", bare, vscode.ConfigurationTarget.Global);
    forgetInterpreter();
    await assert.rejects(
      () => resolvePython(),
      (error: unknown) => {
        assert.ok(
          error instanceof MissingDependencyError,
          `expected MissingDependencyError, got ${String(error)}`,
        );
        assert.deepStrictEqual(error.tried, [bare]);
        return true;
      },
    );
  });

  test("it can name a Python new enough to install into", async function () {
    this.timeout(60000);
    const target = await findInstallTarget();
    assert.ok(target, "no install target found");
    const { stdout } = await run(target, [
      "-c",
      "import sys; print('%d.%d' % sys.version_info[:2])",
    ]);
    const [major, minor] = stdout.trim().split(".").map(Number);
    assert.ok(
      major > 3 || (major === 3 && minor >= 10),
      `install target must be 3.10+, got ${stdout.trim()}`,
    );
  });
});
