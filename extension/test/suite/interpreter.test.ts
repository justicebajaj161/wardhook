import * as assert from "assert";
import { execFile } from "child_process";
import { promisify } from "util";
import { forgetInterpreter, resolvePython } from "../../src/sidecar";

const run = promisify(execFile);

suite("interpreter discovery", () => {
  setup(() => forgetInterpreter());
  suiteTeardown(() => forgetInterpreter());

  test("picks an interpreter that can actually import the library", async () => {
    const python = await resolvePython();
    // The regression this guards: `python3` on macOS often resolves to the
    // system 3.9, which cannot import the library at all. Choosing without
    // probing produced an analyzer that died silently on every scan.
    const { stdout } = await run(python, [
      "-c",
      "import wardhook.guardrails as g; print(g.__version__)",
    ]);
    assert.match(stdout.trim(), /^\d+\.\d+\.\d+$/);
  });

  test("the chosen interpreter is new enough for the library", async () => {
    const python = await resolvePython();
    const { stdout } = await run(python, [
      "-c",
      "import sys; print('%d.%d' % sys.version_info[:2])",
    ]);
    const [major, minor] = stdout.trim().split(".").map(Number);
    assert.ok(
      major > 3 || (major === 3 && minor >= 10),
      `wardhook-guardrails needs Python 3.10+, discovery chose ${stdout.trim()}`,
    );
  });

  test("the result is cached, and forgetInterpreter clears it", async () => {
    const first = await resolvePython();
    assert.strictEqual(await resolvePython(), first);
    forgetInterpreter();
    assert.strictEqual(await resolvePython(), first);
  });
});
