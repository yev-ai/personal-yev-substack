import * as assert from "assert";
import { extensions } from "vscode";

suite("Extension", () => {
  test("Should be present", () => {
    assert.ok(extensions.getExtension("yreztsov.dx"));
  });

  test("Should activate", async () => {
    const ext = extensions.getExtension("yreztsov.dx");
    if (ext) {
      await ext.activate();
      assert.strictEqual(ext.isActive, true);
    }
  });
});
