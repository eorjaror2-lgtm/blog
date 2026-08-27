// runAdComplianceRepair() with a mocked Anthropic client — no real network
// call. Focuses on the deterministic guards (structural, numeric, no-op)
// actually being wired into the repair core in the right order, on top of
// the pure-function tests in ad-compliance-readiness.test.mjs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { runAdComplianceRepair } from "../server.js";
import { makeDraft, makeIssue, createSequentialMockClient } from "./helpers.mjs";

const TOPIC = "유방 석회화, 꼭 조직검사를 해야 할까요?";

test("runAdComplianceRepair: valid minimal fix is accepted", async () => {
  const original = makeDraft();
  const repaired = makeDraft({
    sections: [
      { ...original.sections[0], body: original.sections[0].body.replace(" 안녕유외과가 다른 병원보다 가장 정확합니다.", "") },
      original.sections[1],
    ],
  });
  const mock = createSequentialMockClient([{ parsed_output: repaired }]);
  const result = await runAdComplianceRepair(mock.client, { topic: TOPIC, draft: original, issuesToFix: [makeIssue()] });
  assert.equal(result.ok, true);
  assert.equal(mock.calls.length, 1);
});

test("runAdComplianceRepair (test E): rejects a repair that introduces a new number", async () => {
  const original = makeDraft();
  const repaired = makeDraft({
    conclusion: original.conclusion + " 치료 성공률은 95%에 달합니다.",
  });
  const mock = createSequentialMockClient([{ parsed_output: repaired }]);
  const result = await runAdComplianceRepair(mock.client, { topic: TOPIC, draft: original, issuesToFix: [makeIssue()] });
  assert.equal(result.ok, false);
  assert.equal(result.body.code, "AD_COMPLIANCE_REPAIR_FAILED");
});

test("runAdComplianceRepair (test F): rejects a repair that deletes most of the draft", async () => {
  const original = makeDraft({
    sections: [
      { heading: "섹션 1", body: "첫 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 2", body: "두 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 3", body: "세 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 4", body: "네 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
    ],
  });
  const repaired = { ...original, sections: original.sections.slice(0, 2) }; // dropped 2 of 4 sections
  const mock = createSequentialMockClient([{ parsed_output: repaired }]);
  const result = await runAdComplianceRepair(mock.client, { topic: TOPIC, draft: original, issuesToFix: [makeIssue()] });
  assert.equal(result.ok, false);
  assert.equal(result.body.code, "AD_COMPLIANCE_REPAIR_FAILED");
});

test("runAdComplianceRepair (test H): schema parse failure is a typed, fail-closed error", async () => {
  const original = makeDraft();
  const mock = createSequentialMockClient([{ parsed_output: undefined }]);
  const result = await runAdComplianceRepair(mock.client, { topic: TOPIC, draft: original, issuesToFix: [makeIssue()] });
  assert.equal(result.ok, false);
  assert.equal(result.status, 502);
  assert.equal(result.body.code, "AD_COMPLIANCE_REPAIR_FAILED");
});

test("runAdComplianceRepair: no-op guard rejects an unchanged draft", async () => {
  const original = makeDraft();
  const mock = createSequentialMockClient([{ parsed_output: makeDraft() }]); // byte-identical plain text
  const result = await runAdComplianceRepair(mock.client, { topic: TOPIC, draft: original, issuesToFix: [makeIssue()] });
  assert.equal(result.ok, false);
  assert.equal(result.body.code, "AD_COMPLIANCE_REPAIR_FAILED");
});
