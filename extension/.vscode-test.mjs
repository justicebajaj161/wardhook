import { defineConfig } from "@vscode/test-cli";

export default defineConfig({
  files: "out/test/suite/**/*.test.js",
  version: "stable",
  // The repo root, so the analyzer finds packages/wardhook-guardrails/src on
  // PYTHONPATH without the package having to be pip-installed first.
  workspaceFolder: "..",
  mocha: { ui: "tdd", timeout: 60000 },
});
