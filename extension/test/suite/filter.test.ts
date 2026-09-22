import * as assert from "assert";
import {
  PiiMatch,
  Severity,
  WardhookConfig,
  matchGlob,
  shouldReport,
} from "../../src/config";

function match(entity: string, severity: Severity, validated: boolean): PiiMatch {
  return {
    entity,
    severity,
    validated,
    start: 0,
    end: 1,
    length: 1,
    replacement: `[${entity}]`,
    startLine: 0,
    startChar: 0,
    endLine: 0,
    endChar: 1,
  };
}

function config(overrides: Partial<WardhookConfig> = {}): WardhookConfig {
  return {
    pack: "default",
    customPackPath: "",
    minSeverity: "high",
    validatedOnly: false,
    scanOnType: true,
    debounceMs: 150,
    excludeGlobs: [],
    excludeEntities: [],
    maxFileSizeKb: 1024,
    injectionThreshold: 0.5,
    pythonPath: "",
    policyPackRepo: "",
    policyPackPath: "",
    ...overrides,
  };
}

suite("reporting filter", () => {
  test("keeps high and critical severities", () => {
    assert.strictEqual(shouldReport(match("US_SSN", "high", false), config()), true);
    assert.strictEqual(shouldReport(match("AWS_ACCESS_KEY", "critical", false), config()), true);
  });

  test("drops the medium-severity noise that dominates real source files", () => {
    assert.strictEqual(shouldReport(match("EMAIL", "medium", false), config()), false);
    assert.strictEqual(shouldReport(match("PHONE", "medium", false), config()), false);
  });

  test("a checksum-validated match survives any severity floor", () => {
    const strict = config({ minSeverity: "critical" });
    assert.strictEqual(shouldReport(match("CREDIT_CARD", "medium", true), strict), true);
  });

  test("validatedOnly keeps just the checksum-confirmed matches", () => {
    const strict = config({ validatedOnly: true });
    assert.strictEqual(shouldReport(match("CREDIT_CARD", "critical", true), strict), true);
    assert.strictEqual(shouldReport(match("AWS_ACCESS_KEY", "critical", false), strict), false);
  });

  test("an excluded entity is never reported, even when validated", () => {
    const skip = config({ excludeEntities: ["CREDIT_CARD"] });
    assert.strictEqual(shouldReport(match("CREDIT_CARD", "critical", true), skip), false);
  });

  test("lowering minSeverity surfaces the suppressed matches", () => {
    const loose = config({ minSeverity: "medium" });
    assert.strictEqual(shouldReport(match("EMAIL", "medium", false), loose), true);
  });
});

suite("glob matching", () => {
  test("** spans directories, including none", () => {
    assert.ok(matchGlob("tests/test_pii.py", "**/tests/**"));
    assert.ok(matchGlob("packages/a/tests/test_pii.py", "**/tests/**"));
    assert.ok(!matchGlob("src/pii.py", "**/tests/**"));
  });

  test("* stops at a path separator", () => {
    assert.ok(matchGlob("test_pii.py", "**/test_*.py"));
    assert.ok(matchGlob("a/b/test_pii.py", "**/test_*.py"));
    assert.ok(!matchGlob("test_dir/pii.py", "**/test_*.py"));
  });

  test("dots are literal, not wildcards", () => {
    assert.ok(matchGlob("a.py", "*.py"));
    assert.ok(!matchGlob("axpy", "*.py"));
  });

  test("windows separators are normalised", () => {
    assert.ok(matchGlob("packages\\a\\tests\\x.py", "**/tests/**"));
  });
});
