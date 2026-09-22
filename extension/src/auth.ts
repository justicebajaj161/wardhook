import * as vscode from "vscode";
import { getConfig } from "./config";

const SCOPES = ["repo"];

/**
 * Signs in through the editor's built-in GitHub provider.
 *
 * The OAuth flow, the token and its refresh are the editor's responsibility;
 * this extension never sees a client secret and never stores a token itself.
 * `createIfNone` makes the consent prompt explicit rather than silent.
 */
export async function getSession(
  createIfNone: boolean,
): Promise<vscode.AuthenticationSession | undefined> {
  try {
    return await vscode.authentication.getSession("github", SCOPES, { createIfNone });
  } catch (error) {
    if (createIfNone) {
      void vscode.window.showErrorMessage(`GitHub sign-in failed: ${String(error)}`);
    }
    return undefined;
  }
}

/**
 * Fetches a shared entity pack from a private repository.
 *
 * Teams keep one policy for what counts as sensitive; this pulls that file so
 * every developer scans against the same rules. Only the pack travels, and
 * only inbound -- nothing that was scanned is ever sent anywhere.
 */
export async function syncPolicyPack(storageUri: vscode.Uri): Promise<vscode.Uri | undefined> {
  const config = getConfig();
  const repo = config.policyPackRepo.trim();
  if (!repo) {
    const picked = await vscode.window.showInputBox({
      title: "Sync entity pack from GitHub",
      prompt: "Repository holding the shared entity pack",
      placeHolder: "owner/repo",
      validateInput: (value) =>
        /^[\w.-]+\/[\w.-]+$/.test(value.trim()) ? undefined : "Enter it as owner/repo.",
    });
    if (!picked) {
      return undefined;
    }
    await vscode.workspace
      .getConfiguration("wardhook")
      .update("policyPackRepo", picked.trim(), vscode.ConfigurationTarget.Workspace);
    return syncPolicyPack(storageUri);
  }

  const session = await getSession(true);
  if (!session) {
    return undefined;
  }

  const url = `https://api.github.com/repos/${repo}/contents/${encodeURIComponent(
    config.policyPackPath,
  )}`;
  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${session.accessToken}`,
      Accept: "application/vnd.github.raw+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "wardhook-vscode",
    },
  });
  if (!response.ok) {
    throw new Error(
      `GitHub returned ${response.status} ${response.statusText} for ${repo}/${config.policyPackPath}`,
    );
  }
  const body = await response.text();

  await vscode.workspace.fs.createDirectory(storageUri);
  const target = vscode.Uri.joinPath(storageUri, "policy-pack.yaml");
  await vscode.workspace.fs.writeFile(target, Buffer.from(body, "utf8"));
  await vscode.workspace
    .getConfiguration("wardhook")
    .update("customPackPath", target.fsPath, vscode.ConfigurationTarget.Workspace);
  return target;
}
