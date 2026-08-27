// Pure/deterministic logic — no Anthropic client, no mock, no I/O.
import { test } from "node:test";
import assert from "node:assert/strict";
import { computeAdComplianceReadiness, validateAdComplianceRepairStructure, findNewNumericTokens } from "../server.js";
import { makeDraft, makeIssue, makeReview } from "./helpers.mjs";

test("computeAdComplianceReadiness: clean review -> ready, no human review", () => {
  const review = makeReview();
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adComplianceReady, true);
  assert.equal(gate.requiresHumanReview, false);
  assert.equal(gate.humanReviewReason, null);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.adWarningCount, 0);
});

test("computeAdComplianceReadiness: any blocking issue -> not ready", () => {
  const review = makeReview({ verdict: "needs_revision", issues: [makeIssue()] });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adComplianceReady, false);
  assert.equal(gate.adBlockingCount, 1);
});

test("computeAdComplianceReadiness: warning-only issues do not block readiness", () => {
  const review = makeReview({
    verdict: "pass",
    issues: [makeIssue({ severity: "warning", recommendedAction: "soften" })],
  });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adComplianceReady, true);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.adWarningCount, 1);
});

test("computeAdComplianceReadiness: human_review issue blocks readiness even with zero blocking issues (section 4/8)", () => {
  const review = makeReview({
    verdict: "pass",
    issues: [makeIssue({ severity: "warning", recommendedAction: "human_review" })],
  });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.requiresHumanReview, true);
  assert.equal(gate.adComplianceReady, false, "a human_review-flagged issue must never be auto-resolved into ready:true");
  assert.match(gate.humanReviewReason, /사람의 확인/);
});

test("computeAdComplianceReadiness: confirm_requirement blocks readiness even with zero issues (section 9)", () => {
  const review = makeReview({ priorReviewCheck: "confirm_requirement" });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.requiresHumanReview, true);
  assert.equal(gate.adComplianceReady, false, "repair cannot clear a procedural prior-review requirement — it must never resolve to ready:true on its own");
  assert.match(gate.humanReviewReason, /사전심의/);
});

test("computeAdComplianceReadiness: not_determined + no issues does not force human review", () => {
  const review = makeReview({ priorReviewCheck: "not_determined" });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.requiresHumanReview, false);
  assert.equal(gate.adComplianceReady, true);
});

// --- contradictory verdict guard (Phase 4A-3 final closeout) ---
// computeAdComplianceReadiness() must never derive ready:true from blocking
// count alone when the reviewer's own verdict says needs_revision, and must
// never derive ready:true when verdict says pass but blocking/human-review
// signals say otherwise. Four cases from the closeout brief.

test("contradictory verdict 1: verdict=needs_revision + blocking=0 -> not ready", () => {
  const review = makeReview({
    verdict: "needs_revision",
    issues: [makeIssue({ severity: "warning", recommendedAction: "soften" })],
  });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.adComplianceReady, false, "needs_revision verdict must never be overridden by a zero blocking count");
});

test("contradictory verdict 2: verdict=pass + blocking=0 + no human review -> ready", () => {
  const review = makeReview({ verdict: "pass", issues: [] });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.adBlockingCount, 0);
  assert.equal(gate.requiresHumanReview, false);
  assert.equal(gate.adComplianceReady, true);
});

test("contradictory verdict 3: verdict=pass + human_review issue -> not ready", () => {
  const review = makeReview({
    verdict: "pass",
    issues: [makeIssue({ severity: "warning", recommendedAction: "human_review" })],
  });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.requiresHumanReview, true);
  assert.equal(gate.adComplianceReady, false);
});

test("contradictory verdict 4: verdict=pass + priorReviewCheck=confirm_requirement -> not ready", () => {
  const review = makeReview({ verdict: "pass", priorReviewCheck: "confirm_requirement" });
  const gate = computeAdComplianceReadiness(review);
  assert.equal(gate.requiresHumanReview, true);
  assert.equal(gate.adComplianceReady, false);
});

// --- structural corruption guard (section 13) ---

test("validateAdComplianceRepairStructure: identical section count is fine", () => {
  const original = makeDraft();
  const repaired = makeDraft({ sections: original.sections.map((s) => ({ ...s })) });
  const result = validateAdComplianceRepairStructure(original, repaired);
  assert.equal(result.ok, true);
});

test("validateAdComplianceRepairStructure: rejects section count increase", () => {
  const original = makeDraft();
  const repaired = makeDraft({
    sections: [...original.sections, { heading: "새 섹션", body: "이것은 원래 없던 새로운 section의 본문입니다. 충분히 긴 문장입니다." }],
  });
  const result = validateAdComplianceRepairStructure(original, repaired);
  assert.equal(result.ok, false);
  assert.equal(result.reason, "sectionCountIncreased");
});

test("validateAdComplianceRepairStructure: allows dropping exactly one section", () => {
  const original = makeDraft({
    sections: [
      { heading: "섹션 1", body: "첫 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 2", body: "두 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 3", body: "세 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
    ],
  });
  const repaired = { ...original, sections: original.sections.slice(0, 2) };
  const result = validateAdComplianceRepairStructure(original, repaired);
  assert.equal(result.ok, true);
});

test("validateAdComplianceRepairStructure: rejects dropping more than one section (test F)", () => {
  const original = makeDraft({
    sections: [
      { heading: "섹션 1", body: "첫 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 2", body: "두 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 3", body: "세 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
      { heading: "섹션 4", body: "네 번째 section의 본문입니다. 충분히 길게 작성된 문장입니다." },
    ],
  });
  const repaired = { ...original, sections: original.sections.slice(0, 2) };
  const result = validateAdComplianceRepairStructure(original, repaired);
  assert.equal(result.ok, false);
  assert.equal(result.reason, "sectionCountDroppedTooMuch");
});

test("validateAdComplianceRepairStructure: rejects content shrinking below half the original length", () => {
  const original = makeDraft();
  const repaired = makeDraft({
    introduction: "짧음.",
    sections: original.sections.map((s) => ({ heading: s.heading, body: "짧음." })),
    conclusion: "짧음.",
  });
  const result = validateAdComplianceRepairStructure(original, repaired);
  assert.equal(result.ok, false);
  assert.equal(result.reason, "contentShrankTooMuch");
});

// --- numeric-safety guard (section 14 / test E) ---

test("findNewNumericTokens: no new numbers -> empty", () => {
  const original = makeDraft();
  const repaired = makeDraft({
    sections: [
      { ...original.sections[0], body: original.sections[0].body.replace("안녕유외과가 다른 병원보다 가장 정확합니다.", "") },
      original.sections[1],
    ],
  });
  assert.deepEqual(findNewNumericTokens(original, repaired), []);
});

test("findNewNumericTokens: flags a brand-new number introduced by repair", () => {
  const original = makeDraft();
  const repaired = makeDraft({
    conclusion: original.conclusion + " 치료 성공률은 95%에 달합니다.",
  });
  const found = findNewNumericTokens(original, repaired);
  assert.ok(found.includes("95"), `expected "95" to be flagged as new, got ${JSON.stringify(found)}`);
});

test("findNewNumericTokens: a number already present anywhere in the original is not flagged", () => {
  const original = makeDraft({ introduction: makeDraft().introduction + " BI-RADS 4A 범주입니다." });
  const repaired = makeDraft({
    introduction: original.introduction,
    conclusion: original.conclusion + " BI-RADS 4A 범주라는 점을 다시 안내드립니다.",
  });
  assert.deepEqual(findNewNumericTokens(original, repaired), []);
});
