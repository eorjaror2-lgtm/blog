// Phase 4B-2 — ad repair (inside /api/finalize-evidence-draft's STEP 4,
// runFinalizeAdComplianceStep()) only edits phrasing flagged by the ad
// reviewer; it performs no medical judgment. This exercises the two
// functions handleFinalizeEvidenceDraft() chains together exactly the same
// way: runFinalizeAdComplianceStep() (ad review -> repair at most once ->
// re-review, reused from Phase 4A-3/4B-1 unchanged) and, only when that
// step actually repaired the draft, runFinalMedicalRecheckStep() (the
// existing medical/fact reviewer, reused as-is) run once more as a
// semantic-drift backstop. Mocked Anthropic client only, no real network
// call anywhere in this file.
import { test } from "node:test";
import assert from "node:assert/strict";
import { runFinalizeAdComplianceStep, runFinalMedicalRecheckStep } from "../server.js";
import { makeDraft, makeIssue, makeReview, createSequentialMockClient } from "./helpers.mjs";

const TOPIC = "유방 석회화, 꼭 조직검사를 해야 할까요?";
const MEDICAL_CONTEXT = {
  research: "안녕유외과 관련 근거 dossier 텍스트입니다.",
  evidence: { tier1Sufficient: true, tier2Used: false, missingQuestions: [], optionalGaps: [] },
};

const MEDICAL_PASS_REVIEW = { verdict: "pass", issues: [], summary: "재검토 결과 의학적 의미 변경이 발견되지 않았습니다." };
const MEDICAL_BLOCKING_REVIEW = {
  verdict: "needs_revision",
  issues: [
    {
      severity: "blocking",
      category: "unsupported_claim",
      draftExcerpt: "조직검사가 필요하지 않습니다",
      reason: "근거 dossier와 상충하는 의학적 주장으로 확인됩니다.",
      evidenceBasis: "dossier상 BI-RADS 4A는 여전히 조직검사가 권고되는 범주입니다.",
      recommendedAction: "remove",
    },
  ],
  summary: "광고 수정 과정에서 의학적 의미가 달라진 것으로 확인됩니다.",
};

function draftWithoutFlaggedPhrase(original) {
  return makeDraft({
    sections: [
      { ...original.sections[0], body: original.sections[0].body.replace(" 안녕유외과가 다른 병원보다 가장 정확합니다.", "") },
      original.sections[1],
    ],
  });
}

// --- A: no ad repair -> final medical recheck must never be called ---
test("A: ad review clean (no repair) -> final medical recheck never invoked", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([{ parsed_output: makeReview() }]);
  const step = await runFinalizeAdComplianceStep(mock.client, { medicalFactReady: true, topic: TOPIC, draft });
  assert.equal(step.outcome, "ok");
  assert.equal(step.adRepaired, false);

  // Same conditional handleFinalizeEvidenceDraft() uses: only call the
  // recheck when adRepaired is true.
  if (step.adRepaired) {
    await runFinalMedicalRecheckStep(mock.client, { topic: TOPIC, draft: step.adDraft, ...MEDICAL_CONTEXT });
  }
  assert.equal(mock.calls.length, 1, "no medical recheck call when ad repair did not happen");
});

// --- B: ad repair happens, ad re-review passes, medical recheck passes -> ready preserved ---
test("B: ad repair + ad re-review PASS + medical recheck PASS -> publication-ready preserved", async () => {
  const draft = makeDraft();
  const repairedByAd = draftWithoutFlaggedPhrase(draft);
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) }, // ad initial review
    { parsed_output: repairedByAd }, // ad repair
    { parsed_output: makeReview() }, // ad re-review
    { parsed_output: MEDICAL_PASS_REVIEW }, // final medical recheck
  ]);

  const step = await runFinalizeAdComplianceStep(mock.client, { medicalFactReady: true, topic: TOPIC, draft });
  assert.equal(step.adRepaired, true);
  assert.equal(step.adComplianceReady, true);
  assert.equal(mock.calls.length, 3, "review -> repair -> re-review, exactly 3 ad calls");

  const recheck = await runFinalMedicalRecheckStep(mock.client, { topic: TOPIC, draft: step.adDraft, ...MEDICAL_CONTEXT });
  assert.equal(recheck.outcome, "ok");
  assert.equal(recheck.medicalFactReady, true);
  assert.equal(mock.calls.length, 4, "exactly one additional call for the medical recheck");

  const publicationReady = recheck.medicalFactReady && step.adComplianceReady;
  assert.equal(publicationReady, true);
});

// --- C: ad repair happens, but the final medical recheck finds a problem ---
test("C: ad repair + final medical recheck finds a problem -> medicalFactReady=false, publicationReady=false", async () => {
  const draft = makeDraft();
  const repairedByAd = draftWithoutFlaggedPhrase(draft);
  const mock = createSequentialMockClient([
    { parsed_output: makeReview({ verdict: "needs_revision", issues: [makeIssue()] }) },
    { parsed_output: repairedByAd },
    { parsed_output: makeReview() },
    { parsed_output: MEDICAL_BLOCKING_REVIEW },
  ]);

  const step = await runFinalizeAdComplianceStep(mock.client, { medicalFactReady: true, topic: TOPIC, draft });
  assert.equal(step.adRepaired, true);
  assert.equal(step.adComplianceReady, true, "the ad side itself is clean — the drift is medical, not ad-compliance");

  const recheck = await runFinalMedicalRecheckStep(mock.client, { topic: TOPIC, draft: step.adDraft, ...MEDICAL_CONTEXT });
  assert.equal(recheck.outcome, "ok");
  assert.equal(recheck.medicalFactReady, false, "semantic drift caught by the reused medical reviewer");

  const publicationReady = recheck.medicalFactReady && step.adComplianceReady;
  assert.equal(publicationReady, false);

  // E: no further medical or ad repair call is attempted after the recheck
  // finds a problem — fail closed, no loop.
  assert.equal(mock.calls.length, 4, "recheck never triggers a second medical or ad repair call");
});

// --- D: final medical recheck itself fails technically -> fail-closed error ---
test("D: final medical recheck schema-parse failure -> typed fail-closed error, not a silent ready=false", async () => {
  const draft = makeDraft();
  const mock = createSequentialMockClient([{ parsed_output: null }]);
  const recheck = await runFinalMedicalRecheckStep(mock.client, { topic: TOPIC, draft, ...MEDICAL_CONTEXT });
  assert.equal(recheck.outcome, "fail");
  assert.equal(recheck.status, 502);
  assert.equal(recheck.body.code, "EVIDENCE_DRAFT_REVIEW_FAILED");
});
