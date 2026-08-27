// runAdComplianceFinalizeWorkflow() end to end, with a mocked Anthropic
// client — no real network call anywhere in this file. Covers Phase 4A-3
// brief section 15 cases A, B, C, D, G, I, L (E/F/H are covered directly
// against runAdComplianceRepair() in ad-compliance-repair.test.mjs).
import { test } from "node:test";
import assert from "node:assert/strict";
import { runAdComplianceFinalizeWorkflow } from "../server.js";
import { makeDraft, makeIssue, makeReview, createSequentialMockClient } from "./helpers.mjs";

const INPUT = { publicationChannel: "naver_blog", topic: "유방 석회화, 꼭 조직검사를 해야 할까요?" };

function draftWithoutFlaggedPhrase(original) {
  return makeDraft({
    sections: [
      { ...original.sections[0], body: original.sections[0].body.replace(" 안녕유외과가 다른 병원보다 가장 정확합니다.", "") },
      original.sections[1],
    ],
  });
}

// --- A: initial review passes -> no repair call, ready=true ---
test("case A: clean initial review -> no repair, adComplianceReady=true", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([{ parsed_output: makeReview() }]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "ok");
  assert.equal(result.repaired, false);
  assert.equal(result.adComplianceReady, true);
  assert.equal(result.requiresHumanReview, false);
  assert.equal(mock.calls.length, 1, "no repair or re-review call when the initial review already passes");
});

// --- B: blocking issue -> repair once -> re-review passes -> ready=true ---
test("case B: blocking issue repaired, re-review passes -> adComplianceReady=true", async () => {
  const draft = makeDraft();
  const repaired = draftWithoutFlaggedPhrase(draft);
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: repaired },
    { parsed_output: makeReview() },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "ok");
  assert.equal(result.repaired, true);
  assert.equal(result.adComplianceReady, true);
  assert.equal(mock.calls.length, 3);
});

// --- C / L: blocking issue persists after repair -> no second repair, ready=false ---
test("case C/L: blocking issue survives re-review -> ready=false, repair never attempted twice", async () => {
  const draft = makeDraft();
  const repaired = draftWithoutFlaggedPhrase(draft);
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: repaired },
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue({ draftExcerpt: "여전히 남은 문제 표현" })] }) },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "ok");
  assert.equal(result.repaired, true);
  assert.equal(result.adComplianceReady, false);
  assert.equal(mock.calls.length, 3, "exactly review -> repair -> re-review, never a second repair call");
});

// --- D: human_review issue -> never auto-resolved, repair skipped entirely ---
test("case D: human_review blocking issue -> repair skipped, ready=false, requiresHumanReview=true", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([
    {
      parsed_output: makeReview({
        verdict: "needs_revision",
        priorReviewCheck: "confirm_requirement",
        issues: [makeIssue({ recommendedAction: "human_review" })],
      }),
    },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "ok");
  assert.equal(result.repaired, false, "a human_review-only blocking issue must never trigger an auto-repair attempt");
  assert.equal(result.adComplianceReady, false);
  assert.equal(result.requiresHumanReview, true);
  assert.equal(mock.calls.length, 1, "repair and re-review must never be called when nothing is auto-fixable");
});

// --- G: repair silently introduces new promotional content -> caught by re-review, not by a heuristic ---
test("case G: repair-introduced promotional phrase is caught by re-review, not silently passed", async () => {
  const draft = makeDraft();
  const repairedWithNewPromo = makeDraft({
    sections: [
      { ...draft.sections[0], body: draft.sections[0].body.replace(" 안녕유외과가 다른 병원보다 가장 정확합니다.", "") },
      draft.sections[1],
    ],
    conclusion: draft.conclusion + " 지금 바로 안녕유외과를 방문해주세요.",
  });
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: repairedWithNewPromo },
    {
      parsed_output: makeReview({
        verdict: "needs_revision",
        issues: [makeIssue({ draftExcerpt: "지금 바로 안녕유외과를 방문해주세요.", reason: "특정 의료기관 방문을 직접 유도하는 표현입니다." })],
      }),
    },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "ok");
  assert.equal(result.adComplianceReady, false, "re-review must catch the new issue even though structural/numeric guards had nothing to flag");
  assert.equal(mock.calls.length, 3, "still exactly one repair attempt, never a second one to fix the new issue");
});

// --- H: repair schema-parse failure inside the workflow -> hard fail, no third call ---
test("case H: repair schema-parse failure inside the workflow stops before re-review", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: undefined },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "fail");
  assert.equal(mock.calls.length, 2, "re-review must never be attempted after a failed repair");
});

// --- I: second review (re-review) fails -> never returned as ready ---
test("case I: re-review API failure -> workflow fails closed, never reports ready", async () => {
  const draft = makeDraft();
  const repaired = draftWithoutFlaggedPhrase(draft);
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: repaired },
    { throwError: new Error("simulated Anthropic API failure") },
  ]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "fail");
  assert.equal("adComplianceReady" in result, false, "a failed re-review must never carry a readiness verdict of any kind");
  assert.equal(mock.calls.length, 3);
});

// --- initial review technical failure also fails closed, no repair attempted ---
test("initial review API failure -> workflow fails closed immediately", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([{ throwError: new Error("simulated Anthropic API failure") }]);
  const result = await runAdComplianceFinalizeWorkflow(mock.client, { ...INPUT, draft });
  assert.equal(result.outcome, "fail");
  assert.equal(mock.calls.length, 1);
});
