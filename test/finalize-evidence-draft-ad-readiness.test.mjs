// Phase 4B-1 closeout — runFinalizeAdComplianceStep() (the ad-compliance
// STEP 4 inside /api/finalize-evidence-draft) used to compute its own
// adComplianceReady: adBlockingCount === 0, ignoring verdict and
// requiresHumanReview entirely. It now reuses computeAdComplianceReadiness()
// — the same gate the standalone /api/finalize-ad-compliance endpoint uses
// (see ad-compliance-finalize-workflow.test.mjs) — so both endpoints agree
// on the same review. Mocked Anthropic client only, no real network call.
import { test } from "node:test";
import assert from "node:assert/strict";
import { runFinalizeAdComplianceStep } from "../server.js";
import { makeDraft, makeIssue, makeReview, createSequentialMockClient } from "./helpers.mjs";

const INPUT = { medicalFactReady: true, topic: "유방 석회화, 꼭 조직검사를 해야 할까요?", draft: makeDraft() };

test("runFinalizeAdComplianceStep: verdict=pass, blocking=0, no human review -> adComplianceReady=true", async () => {
  const mock = createSequentialMockClient([{ parsed_output: makeReview() }]);
  const step = await runFinalizeAdComplianceStep(mock.client, INPUT);
  assert.equal(step.outcome, "ok");
  assert.equal(step.adBlockingCount, 0);
  assert.equal(step.requiresHumanReview, false);
  assert.equal(step.adComplianceReady, true);
});

test("runFinalizeAdComplianceStep: verdict=needs_revision, blocking=0 -> adComplianceReady=false", async () => {
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue({ severity: "warning", recommendedAction: "soften" })] }) },
  ]);
  const step = await runFinalizeAdComplianceStep(mock.client, INPUT);
  assert.equal(step.outcome, "ok");
  assert.equal(step.adBlockingCount, 0, "needs_revision here is carried only by a warning issue, never a blocking one");
  assert.equal(
    step.adComplianceReady,
    false,
    "a needs_revision verdict must never be overridden by a zero blocking count — this is exactly the gap the old adBlockingCount===0 computation had",
  );
});

test("runFinalizeAdComplianceStep: human_review issue present -> adComplianceReady=false", async () => {
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "pass", issues: [makeIssue({ severity: "warning", recommendedAction: "human_review" })] }) },
  ]);
  const step = await runFinalizeAdComplianceStep(mock.client, INPUT);
  assert.equal(step.outcome, "ok");
  assert.equal(step.adBlockingCount, 0);
  assert.equal(step.requiresHumanReview, true);
  assert.equal(step.adComplianceReady, false, "the old adBlockingCount===0 computation ignored human_review entirely");
});

test("runFinalizeAdComplianceStep: priorReviewCheck=confirm_requirement -> adComplianceReady=false", async () => {
  const mock = createSequentialMockClient([{ parsed_output: makeReview({ verdict: "pass", priorReviewCheck: "confirm_requirement" }) }]);
  const step = await runFinalizeAdComplianceStep(mock.client, INPUT);
  assert.equal(step.outcome, "ok");
  assert.equal(step.adBlockingCount, 0);
  assert.equal(step.requiresHumanReview, true);
  assert.equal(step.adComplianceReady, false, "the old adBlockingCount===0 computation ignored priorReviewCheck entirely");
});

test("runFinalizeAdComplianceStep: medicalFactReady=false -> skipped, no Anthropic call", async () => {
  const mock = createSequentialMockClient([]);
  const step = await runFinalizeAdComplianceStep(mock.client, { ...INPUT, medicalFactReady: false });
  assert.equal(step.outcome, "skip");
  assert.equal(mock.calls.length, 0);
});
