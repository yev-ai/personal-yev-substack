import { defineConfig } from "@vscode/test-cli";

export default defineConfig({
  files: "out/**/*.test.js",
  launchArgs: ["--log", "error"],
});
