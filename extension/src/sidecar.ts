import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { getConfig } from "./config";

/** A request that has been written to the pipe and is awaiting its response. */
interface Pending {
  resolve: (value: any) => void;
  reject: (reason: Error) => void;
}

const PROBE = "import wardhook.guardrails as g; g.PIIDetector().detect('probe')";

export class SidecarError extends Error {}

/**
 * Owns the one long-lived Python analyzer process.
 *
 * Importing `wardhook.guardrails` costs about 160ms. Paying that per keystroke
 * is unusable; paying it once per session and then round-tripping over a pipe
 * costs about 12ms for a 38KB file. That trade is the whole reason this class
 * exists rather than a `spawnSync` per scan.
 *
 * The process is started lazily on the first request and restarted on demand
 * after a crash, so a malformed document cannot permanently disable scanning.
 */
export class Sidecar implements vscode.Disposable {
  private proc: ChildProcessWithoutNullStreams | undefined;
  private starting: Promise<void> | undefined;
  private readonly pending = new Map<number, Pending>();
  private buffer = "";
  private nextId = 1;
  private disposed = false;
  private consecutiveFailures = 0;

  constructor(
    private readonly scriptPath: string,
    private readonly log: vscode.OutputChannel,
  ) {}

  /** Sends one request and resolves with the handler's result. */
  async request<T = any>(
    op: string,
    payload: Record<string, unknown> = {},
    token?: vscode.CancellationToken,
  ): Promise<T> {
    if (this.disposed) {
      throw new SidecarError("The Wardhook analyzer has been shut down.");
    }
    await this.ensureStarted();
    const proc = this.proc;
    if (!proc) {
      throw new SidecarError("The Wardhook analyzer is not running.");
    }
    const id = this.nextId++;
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      if (token) {
        // The analyzer handles one request at a time, so a cancelled request
        // is abandoned rather than interrupted: we stop waiting, and its
        // response is discarded when it arrives.
        token.onCancellationRequested(() => {
          if (this.pending.delete(id)) {
            reject(new vscode.CancellationError());
          }
        });
      }
    });
    proc.stdin.write(`${JSON.stringify({ id, op, ...payload })}\n`);
    return promise;
  }

  /** Starts the process if it is not already running, coalescing callers. */
  private ensureStarted(): Promise<void> {
    if (this.proc && !this.proc.killed) {
      return Promise.resolve();
    }
    if (!this.starting) {
      this.starting = this.start().finally(() => {
        this.starting = undefined;
      });
    }
    return this.starting;
  }

  private async start(): Promise<void> {
    const python = await resolvePython();
    const env = { ...process.env, PYTHONUNBUFFERED: "1", ...developmentPythonPath() };
    this.log.appendLine(`[sidecar] interpreter: ${python}`);
    this.log.appendLine(`[sidecar] starting: ${python} -u ${this.scriptPath}`);

    const proc = spawn(python, ["-u", this.scriptPath], { env });
    this.proc = proc;
    this.buffer = "";

    proc.stdout.setEncoding("utf8");
    proc.stdout.on("data", (chunk: string) => this.onData(chunk));
    proc.stderr.setEncoding("utf8");
    proc.stderr.on("data", (chunk: string) => this.log.appendLine(`[python] ${chunk.trimEnd()}`));
    proc.on("exit", (code, signal) => this.onExit(code, signal));
    proc.on("error", (err) => {
      this.log.appendLine(`[sidecar] spawn failed: ${err.message}`);
      this.failAll(new SidecarError(`Could not start Python (${python}): ${err.message}`));
    });

    // The analyzer announces itself once the library is imported. Waiting for
    // that here means the first real request never races the import.
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new SidecarError("The Wardhook analyzer did not start within 20 seconds."));
      }, 20000);
      const onReady = (chunk: string) => {
        if (!chunk.includes('"ready"')) {
          return;
        }
        clearTimeout(timer);
        proc.stdout.off("data", onReady);
        resolve();
      };
      proc.stdout.on("data", onReady);
      proc.once("exit", () => {
        clearTimeout(timer);
        reject(new SidecarError(importHint(python)));
      });
    });
    this.consecutiveFailures = 0;
    this.log.appendLine("[sidecar] ready");
  }

  /** Splits the stream into lines and settles whichever request each answers. */
  private onData(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline !== -1) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      newline = this.buffer.indexOf("\n");
      if (line) {
        this.onLine(line);
      }
    }
  }

  private onLine(line: string): void {
    let message: any;
    try {
      message = JSON.parse(line);
    } catch {
      this.log.appendLine(`[sidecar] unparseable line: ${line.slice(0, 200)}`);
      return;
    }
    if (message.event === "ready" || typeof message.id !== "number") {
      return;
    }
    const waiter = this.pending.get(message.id);
    if (!waiter) {
      return; // Cancelled before the answer arrived.
    }
    this.pending.delete(message.id);
    if (message.ok) {
      waiter.resolve(message.result);
    } else {
      waiter.reject(new SidecarError(String(message.error ?? "unknown analyzer error")));
    }
  }

  private onExit(code: number | null, signal: string | null): void {
    this.proc = undefined;
    if (this.disposed) {
      return;
    }
    this.consecutiveFailures += 1;
    this.log.appendLine(`[sidecar] exited (code=${code}, signal=${signal})`);
    this.failAll(new SidecarError("The Wardhook analyzer stopped unexpectedly."));
    if (this.consecutiveFailures >= 3) {
      void vscode.window
        .showErrorMessage(
          "The Wardhook analyzer keeps stopping. Check the log for details.",
          "Show Log",
        )
        .then((choice) => {
          if (choice === "Show Log") {
            this.log.show();
          }
        });
    }
  }

  private failAll(error: Error): void {
    for (const waiter of this.pending.values()) {
      waiter.reject(error);
    }
    this.pending.clear();
  }

  /** Stops the process so the next request starts a fresh one. */
  restart(): void {
    this.consecutiveFailures = 0;
    forgetInterpreter();
    this.proc?.kill();
    this.proc = undefined;
  }

  dispose(): void {
    this.disposed = true;
    this.failAll(new SidecarError("The Wardhook analyzer has been shut down."));
    this.proc?.kill();
    this.proc = undefined;
  }
}

/**
 * Finds a Python interpreter that can actually run the analyzer.
 *
 * Picking one is not enough: `python3` on a Mac frequently resolves to the
 * system 3.9, which cannot even import the library (`dataclass(slots=)` needs
 * 3.10) and has no PyYAML. So each candidate is probed by importing the
 * package for real, and the first that succeeds is cached for the session.
 */
export async function resolvePython(): Promise<string> {
  if (cachedInterpreter) {
    return cachedInterpreter;
  }
  const env = { ...process.env, ...developmentPythonPath() };
  const candidates = await interpreterCandidates();
  const tried: string[] = [];
  for (const candidate of candidates) {
    if (tried.includes(candidate)) {
      continue;
    }
    tried.push(candidate);
    if (await canImportGuardrails(candidate, env)) {
      cachedInterpreter = candidate;
      return candidate;
    }
  }
  throw new MissingDependencyError(
    "wardhook-guardrails is not installed for any Python this extension can find.",
    await findInstallTarget(),
    tried,
  );
}

/** Forgets the cached interpreter, e.g. when the setting changes. */
export function forgetInterpreter(): void {
  cachedInterpreter = undefined;
}

let cachedInterpreter: string | undefined;

/** Candidate interpreters, best first. */
async function interpreterCandidates(): Promise<string[]> {
  const candidates: string[] = [];

  const configured = getConfig().pythonPath.trim();
  if (configured) {
    // An explicit setting is a decision, not a hint: try it and nothing else.
    return [configured];
  }

  // A virtualenv beside the code is almost always the right answer, and inside
  // this monorepo it is the only one with the package installed.
  candidates.push(...findVirtualenvs());

  try {
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (ext) {
      const api = ext.isActive ? ext.exports : await ext.activate();
      const folder = vscode.workspace.workspaceFolders?.[0]?.uri;
      const selected = api?.environments?.getActiveEnvironmentPath(folder);
      const resolved = selected
        ? await api.environments.resolveEnvironment(selected)
        : undefined;
      const execPath = resolved?.executable?.uri?.fsPath ?? selected?.path;
      if (execPath) {
        candidates.push(execPath);
      }
    }
  } catch {
    // The Python extension is optional.
  }

  if (process.platform === "win32") {
    candidates.push("python", "py");
  } else {
    // Homebrew and pyenv shims usually precede /usr/bin on PATH, but name the
    // common install locations explicitly in case they do not.
    candidates.push(
      "python3",
      "/opt/homebrew/bin/python3",
      "/usr/local/bin/python3",
      "python3.13",
      "python3.12",
      "python3.11",
      "python3.10",
    );
  }
  return candidates.filter((c) => c.length > 0);
}

/** Virtualenvs found beside, or above, the open folders. */
function findVirtualenvs(): string[] {
  const binary = process.platform === "win32"
    ? path.join("Scripts", "python.exe")
    : path.join("bin", "python");
  const found: string[] = [];
  for (const root of vscode.workspace.workspaceFolders ?? []) {
    let dir = root.uri.fsPath;
    for (let depth = 0; depth < 5; depth++) {
      for (const name of [".venv", "venv", ".virtualenv"]) {
        const candidate = path.join(dir, name, binary);
        if (fs.existsSync(candidate)) {
          found.push(candidate);
        }
      }
      const parent = path.dirname(dir);
      if (parent === dir) {
        break;
      }
      dir = parent;
    }
  }
  return found;
}

/** Runs a candidate and reports whether the library imports. */
function canImportGuardrails(python: string, env: NodeJS.ProcessEnv): Promise<boolean> {
  return new Promise((resolve) => {
    let child: ChildProcessWithoutNullStreams;
    try {
      // Importing the package is not enough: PyYAML is pulled in lazily when
      // the default pack is loaded, so an interpreter missing it imports
      // cleanly and then fails on the first real scan. Load a pack here.
      child = spawn(python, ["-c", PROBE], { env });
    } catch {
      resolve(false);
      return;
    }
    const timer = setTimeout(() => {
      child.kill();
      resolve(false);
    }, 15000);
    child.on("error", () => {
      clearTimeout(timer);
      resolve(false);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      resolve(code === 0);
    });
  });
}

/**
 * Adds the monorepo's guardrails source to PYTHONPATH when present.
 *
 * Inside a checkout of Wardhook itself the package is often not installed, and
 * demanding `pip install` before the extension does anything would be a poor
 * first run. Outside such a checkout this contributes nothing.
 */
/** Thrown when no interpreter on the machine can load the library. */
export class MissingDependencyError extends SidecarError {
  constructor(
    message: string,
    readonly installTarget: string | undefined,
    readonly tried: string[],
  ) {
    super(message);
  }
}

/** Returns a candidate's `(major, minor)` version, or undefined if it fails. */
function probeVersion(python: string): Promise<[number, number] | undefined> {
  return new Promise((resolve) => {
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(python, ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"]);
    } catch {
      resolve(undefined);
      return;
    }
    let out = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      out += chunk;
    });
    const timer = setTimeout(() => {
      child.kill();
      resolve(undefined);
    }, 10000);
    child.on("error", () => {
      clearTimeout(timer);
      resolve(undefined);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      const match = /(\d+)\.(\d+)/.exec(out.trim());
      resolve(code === 0 && match ? [Number(match[1]), Number(match[2])] : undefined);
    });
  });
}

/**
 * Picks the best interpreter to install into.
 *
 * The library needs 3.10 or newer, so an interpreter that merely exists is not
 * a valid target -- offering to install into the system 3.9 would fail in a
 * way that looks like the extension's fault.
 */
export async function findInstallTarget(): Promise<string | undefined> {
  for (const candidate of await interpreterCandidates()) {
    const version = await probeVersion(candidate);
    if (version && (version[0] > 3 || (version[0] === 3 && version[1] >= 10))) {
      return candidate;
    }
  }
  return undefined;
}

/** Installs wardhook-guardrails into the given interpreter. */
export function installGuardrails(
  python: string,
  log: vscode.OutputChannel,
): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    log.appendLine(`[install] ${python} -m pip install --upgrade wardhook-guardrails`);
    const child = spawn(python, ["-m", "pip", "install", "--upgrade", "wardhook-guardrails"]);
    let output = "";
    const collect = (chunk: Buffer) => {
      const text = chunk.toString();
      output += text;
      log.appendLine(`[pip] ${text.trimEnd()}`);
    };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    child.on("error", (err) => resolve({ ok: false, output: err.message }));
    child.on("exit", (code) => resolve({ ok: code === 0, output }));
  });
}

export function developmentPythonPath(): Record<string, string> {
  const roots = vscode.workspace.workspaceFolders ?? [];
  const found = new Set<string>();
  for (const root of roots) {
    // Walk up a few levels: the open folder may be a subdirectory of the
    // checkout, such as the demo workspace this extension ships.
    let dir = root.uri.fsPath;
    for (let depth = 0; depth < 5; depth++) {
      const candidate = path.join(dir, "packages", "wardhook-guardrails", "src");
      if (fs.existsSync(path.join(candidate, "wardhook", "guardrails", "__init__.py"))) {
        found.add(path.normalize(candidate));
        break;
      }
      const parent = path.dirname(dir);
      if (parent === dir) {
        break;
      }
      dir = parent;
    }
  }
  if (found.size === 0) {
    return {};
  }
  const parts = [...found];
  const existing = process.env.PYTHONPATH;
  const joined = existing ? [...parts, existing].join(path.delimiter) : parts.join(path.delimiter);
  return { PYTHONPATH: joined };
}

function importHint(python: string): string {
  return (
    `The Wardhook analyzer could not start with ${python}. ` +
    "It needs the wardhook-guardrails package: pip install wardhook-guardrails"
  );
}
