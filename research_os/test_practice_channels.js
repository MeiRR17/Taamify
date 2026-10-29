// Run the actual page's pure rendering functions against a minimal DOM sink.
// This is not a browser/visual test.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const html = fs.readFileSync(path.join(__dirname, "../webapp/static/practice.html"), "utf8");
const context = vm.createContext({
  box: {innerHTML: ""},
  CH_LABELS: {stress: "הטעמה", dagesh: "דגש קל", guttural: "ע׳ / א׳", sheva: "שווא נע/נח"},
});
vm.runInContext("const $ = () => box;", context);
const mastery = html.slice(html.indexOf("function readingMastered("), html.indexOf("function showVerdict("));
const rendering = html.slice(html.indexOf("function renderChannels("), html.indexOf('$("rec").addEventListener'));
vm.runInContext(mastery + rendering, context);

function result(status, mode = "candidate") {
  return {checked: 1, correct: 1, channel_calibration: {mode},
    words: [{status: "ok", channels: {stress: {status}}}],
    channel_summary: {stress: {checked: 1, correct: status === "ok" ? 1 : 0}}};
}
assert.equal(context.readingMastered(result("unsure")), false);
assert.equal(context.readingMastered(result("wrong")), false);
assert.equal(context.readingMastered(result("ok")), true);
assert.equal(context.readingMastered(result("unsure", "legacy")), false);
const partial = result("ok"); partial.words.push({status: "unknown"});
assert.equal(context.readingMastered(partial), false);
assert.equal(context.readingMastered(result("unavailable")), false);
context.renderChannels(result("unsure"));
assert.match(context.box.innerHTML, /ch none/);
assert.match(context.box.innerHTML, /1 לא בטוח/);
assert.doesNotMatch(context.box.innerHTML, /ch bad/);
context.renderChannels(result("wrong"));
assert.match(context.box.innerHTML, /ch bad/);
assert.match(context.box.innerHTML, /1 לתרגול/);
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
console.log("Practice candidate uncertainty, mastery, warnings and script syntax: passed");
