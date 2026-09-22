// Bundles the extension into a single CommonJS file for the VS Code host.
//
// `vscode` is provided by the host at runtime and must never be bundled. The
// Python sidecar is shipped as a plain file and read from disk, so it is not
// an entry point here.
const esbuild = require("esbuild");

const production = process.argv.includes("--production");
const watch = process.argv.includes("--watch");

/** Reports bundle failures as clickable paths rather than a silent exit. */
const problemMatcher = {
  name: "problem-matcher",
  setup(build) {
    build.onEnd((result) => {
      for (const { text, location } of result.errors) {
        console.error(`✘ [ERROR] ${text}`);
        if (location) {
          console.error(`    ${location.file}:${location.line}:${location.column}`);
        }
      }
      console.log(`[${watch ? "watch" : "build"}] done (${result.errors.length} errors)`);
    });
  },
};

async function main() {
  const ctx = await esbuild.context({
    entryPoints: ["src/extension.ts"],
    bundle: true,
    format: "cjs",
    platform: "node",
    target: "node18",
    outfile: "dist/extension.js",
    external: ["vscode"],
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    logLevel: "silent",
    plugins: [problemMatcher],
  });
  if (watch) {
    await ctx.watch();
  } else {
    await ctx.rebuild();
    await ctx.dispose();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
