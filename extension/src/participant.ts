import * as vscode from "vscode";
import { PiiMatch, getConfig, shouldReport } from "./config";
import { Analyzer } from "./diagnostics";
import { Sidecar } from "./sidecar";

const PARTICIPANT_ID = "wardhook.chat";

/** The tools the model may call when answering a free-form question. */
const TOOL_NAMES = [
  "wardhook_scan_pii",
  "wardhook_redact",
  "wardhook_scan_workspace",
  "wardhook_score_injection",
  "wardhook_list_packs",
  "wardhook_explain_entity",
  "wardhook_check_tool_policy",
];

const SYSTEM_PROMPT = [
  "You are Wardhook, an assistant for finding and removing personal data in code.",
  "Use the provided tools rather than guessing; you cannot detect PII by eye reliably.",
  "The detector never returns the matched value, only its position, so never claim to",
  "show the user the secret itself. Be concise and cite file and line numbers.",
].join(" ");

/**
 * Registers the `@wardhook` chat participant.
 *
 * Returns undefined on an editor without the chat API, which is what makes the
 * extension safe to load in a VS Code without Copilot, or in a fork.
 */
export function registerParticipant(
  context: vscode.ExtensionContext,
  sidecar: Sidecar,
  analyzer: Analyzer,
): vscode.Disposable | undefined {
  const chat = vscode.chat as typeof vscode.chat | undefined;
  if (!chat?.createChatParticipant) {
    return undefined;
  }

  const handler: vscode.ChatRequestHandler = async (request, chatContext, stream, token) => {
    switch (request.command) {
      case "scan":
        return handleScan(sidecar, analyzer, stream, token);
      case "redact":
        return handleRedact(sidecar, stream, token);
      case "packs":
        return handlePacks(sidecar, stream, token);
      case "policy":
        return handlePolicy(stream);
      default:
        return handleFreeform(request, chatContext, stream, token);
    }
  };

  const participant = chat.createChatParticipant(PARTICIPANT_ID, handler);
  participant.iconPath = vscode.Uri.joinPath(context.extensionUri, "resources", "shield.svg");
  return participant;
}

async function handleScan(
  sidecar: Sidecar,
  analyzer: Analyzer,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    stream.markdown("Open a file first, or ask me to scan the whole workspace.");
    return;
  }
  stream.progress(`Scanning ${vscode.workspace.asRelativePath(editor.document.uri)}…`);
  const config = getConfig(editor.document.uri);
  const res = await sidecar.request<{ matches: PiiMatch[] }>(
    "scan",
    { text: editor.document.getText(), pack: config.pack },
    token,
  );
  if (res.matches.length === 0) {
    stream.markdown("No personal data detected. ✅");
    return;
  }
  // Report against the same filter the editor uses, so the chat answer and the
  // squiggles can never disagree.
  const reported = res.matches.filter((match) => shouldReport(match, config));
  const suppressed = res.matches.filter((match) => !shouldReport(match, config));

  const line = (match: PiiMatch) => {
    const position = new vscode.Position(match.startLine, match.startChar);
    stream.markdown(
      `- \`${match.entity}\` — ${match.severity}${match.validated ? ", validated" : ""} — `,
    );
    stream.anchor(new vscode.Location(editor.document.uri, position), `line ${match.startLine + 1}`);
    stream.markdown("\n");
  };

  if (reported.length === 0) {
    stream.markdown(
      `Nothing meets the current threshold, though ${res.matches.length} item(s) were detected ` +
        `below it.\n\n`,
    );
  } else {
    stream.markdown(`Reporting **${reported.length}** of ${res.matches.length} detected item(s):\n\n`);
    reported.forEach(line);
  }

  if (suppressed.length > 0) {
    stream.markdown(
      `\n<details><summary>${suppressed.length} suppressed below \`${config.minSeverity}\` severity` +
        `</summary>\n\n`,
    );
    suppressed.forEach(line);
    stream.markdown(
      `\nLower \`wardhook.minSeverity\` to report these.\n\n</details>\n`,
    );
  }
  stream.markdown("\nThe values themselves are never returned by the detector.\n");
  stream.button({
    command: "wardhook.redactFile",
    title: "Redact this file",
  });
  void analyzer;
}

async function handleRedact(
  sidecar: Sidecar,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    stream.markdown("Open a file first.");
    return;
  }
  const selection = editor.selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(editor.selection);
  const res = await sidecar.request<{ text: string; matchCount: number }>(
    "redact",
    { text: selection, pack: getConfig(editor.document.uri).pack },
    token,
  );
  if (res.matchCount === 0) {
    stream.markdown("Nothing to redact.");
    return;
  }
  stream.markdown(`Redacted **${res.matchCount}** item(s):\n\n`);
  stream.markdown(`\`\`\`${editor.document.languageId}\n${res.text}\n\`\`\``);
}

async function handlePacks(
  sidecar: Sidecar,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const res = await sidecar.request<{
    packs: { name: string; entities: string[]; ruleCount: number }[];
  }>("packs", {}, token);
  const active = getConfig().pack;
  stream.markdown("| Pack | Rules | Entities |\n| --- | --- | --- |\n");
  for (const pack of res.packs) {
    stream.markdown(
      `| ${pack.name}${pack.name === active ? " **(active)**" : ""} | ${pack.ruleCount} | ` +
        `${pack.entities.join(", ")} |\n`,
    );
  }
}

function handlePolicy(stream: vscode.ChatResponseStream): void {
  stream.markdown(
    "Ask me in plain language, for example:\n\n" +
      "> Using the policy `{\"support\": [\"lookup_*\"]}`, may the `support` role call `issue_refund`?\n\n" +
      "I will check it against the deny-by-default RBAC rules.",
  );
}

/** Answers anything else by letting the model drive the tools. */
async function handleFreeform(
  request: vscode.ChatRequest,
  chatContext: vscode.ChatContext,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const models = await usableModels(request);
  if (models.length === 0) {
    stream.markdown("No language model is available. Try `/scan`, `/redact` or `/packs`.");
    return;
  }
  let lastError: unknown;
  for (const model of models) {
    try {
      await converse(model, request, chatContext, stream, token);
      return;
    } catch (error) {
      if (error instanceof vscode.CancellationError) {
        throw error;
      }
      lastError = error;
      // A model the extension is not permitted to drive is a routing problem,
      // not a failure: try the next one.
    }
  }
  stream.markdown(
    `I could not reach a language model (${String(lastError)}). ` +
      "The `/scan`, `/redact` and `/packs` commands work without one.",
  );
}

/**
 * Models this extension may actually drive, best first.
 *
 * `request.model` can be Copilot's "auto" selector, which routes on the user's
 * behalf and which an extension is not allowed to use -- asking for it fails
 * with "Language model 'copilot/auto' cannot be used by ...". A concretely
 * selected model is fine, so the auto entry is skipped and real ones follow.
 */
async function usableModels(request: vscode.ChatRequest): Promise<vscode.LanguageModelChat[]> {
  const models: vscode.LanguageModelChat[] = [];
  const requested = request.model;
  if (requested && !isAutoSelector(requested)) {
    models.push(requested);
  }
  try {
    for (const model of await vscode.lm.selectChatModels({ vendor: "copilot" })) {
      if (!isAutoSelector(model) && !models.some((m) => m.id === model.id)) {
        models.push(model);
      }
    }
  } catch {
    // Selection can fail before the user has consented; the loop handles it.
  }
  return models;
}

function isAutoSelector(model: vscode.LanguageModelChat): boolean {
  return /(^|[/-])auto$/i.test(model.id) || /^auto$/i.test(model.family ?? "");
}

/** One tool-calling conversation against a single model. */
async function converse(
  model: vscode.LanguageModelChat,
  request: vscode.ChatRequest,
  chatContext: vscode.ChatContext,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const tools = vscode.lm.tools.filter((tool) => TOOL_NAMES.includes(tool.name));
  const messages = [
    vscode.LanguageModelChatMessage.User(SYSTEM_PROMPT),
    ...historyOf(chatContext),
    vscode.LanguageModelChatMessage.User(request.prompt),
  ];

  // Several passes: the model asks for tools, we run them, then it answers.
  for (let round = 0; round < 3; round++) {
    const response = await model.sendRequest(messages, { tools }, token);
    const calls: vscode.LanguageModelToolCallPart[] = [];
    let said = false;
    for await (const part of response.stream) {
      if (part instanceof vscode.LanguageModelTextPart) {
        stream.markdown(part.value);
        said = true;
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        calls.push(part);
      }
    }
    if (calls.length === 0) {
      if (!said) {
        stream.markdown("I could not produce an answer. Try `/scan` for a direct result.");
      }
      return;
    }
    messages.push(vscode.LanguageModelChatMessage.Assistant(calls));
    const results: vscode.LanguageModelToolResultPart[] = [];
    for (const call of calls) {
      stream.progress(`Running ${call.name}…`);
      try {
        const invoked = await vscode.lm.invokeTool(
          call.name,
          { input: call.input, toolInvocationToken: request.toolInvocationToken },
          token,
        );
        results.push(new vscode.LanguageModelToolResultPart(call.callId, invoked.content));
      } catch (error) {
        results.push(
          new vscode.LanguageModelToolResultPart(call.callId, [
            new vscode.LanguageModelTextPart(`Tool failed: ${String(error)}`),
          ]),
        );
      }
    }
    messages.push(vscode.LanguageModelChatMessage.User(results));
  }
}

/** Replays earlier turns so follow-up questions keep their context. */
function historyOf(chatContext: vscode.ChatContext): vscode.LanguageModelChatMessage[] {
  const messages: vscode.LanguageModelChatMessage[] = [];
  for (const turn of chatContext.history) {
    if (turn instanceof vscode.ChatRequestTurn) {
      messages.push(vscode.LanguageModelChatMessage.User(turn.prompt));
    } else if (turn instanceof vscode.ChatResponseTurn) {
      const text = turn.response
        .filter((part): part is vscode.ChatResponseMarkdownPart => "value" in part)
        .map((part) => part.value.value)
        .join("");
      if (text) {
        messages.push(vscode.LanguageModelChatMessage.Assistant(text));
      }
    }
  }
  return messages;
}
