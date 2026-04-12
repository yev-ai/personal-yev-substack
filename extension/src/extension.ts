import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  console.log("DX activated");
  const disposable = vscode.commands.registerCommand("dx.helloWorld", () => {
    vscode.window.showInformationMessage("Hello: DX");
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
