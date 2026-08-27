// Shared test fixtures/mocks for the Phase 4A-3 ad compliance finalize
// suite. No real Anthropic call anywhere in this file or its callers — every
// test mocks `client.messages.parse` directly. Node's built-in test runner
// (`node --test`) is used; no test framework was added as a dependency.
import { AD_COMPLIANCE_POLICY_PACK } from "../server.js";

// A real ruleId from the live policy pack, so validateAdComplianceReviewSemantics()
// inside runAdComplianceReview() (which checks every issue.ruleId against
// AD_COMPLIANCE_RULE_IDS, derived from this same pack) accepts our fixtures.
// "comparative_claim" is never specially downgraded by
// downgradeInformationBoundaryBlockingIssues() (that only touches
// information_vs_advertising_boundary), so a blocking issue on this ruleId
// stays blocking through normalization — needed for tests that exercise real
// blocking behavior.
export function findRuleId(id) {
  const rule = AD_COMPLIANCE_POLICY_PACK.contentRules.find((r) => r.id === id);
  if (!rule) throw new Error(`fixture ruleId not found in policy pack: ${id}`);
  return rule.id;
}

export const COMPARATIVE_CLAIM_RULE_ID = findRuleId("comparative_claim");

// A structurally complete draft satisfying validateEvidenceDraftContent()'s
// minimums (sections.length >= 2, each heading/body long enough, non-empty
// title/introduction/conclusion) so repair-path completeness checks never
// fail for reasons unrelated to what a given test is actually exercising.
export function makeDraft(overrides = {}) {
  return {
    title: "유방 석회화, 꼭 조직검사를 해야 할까요?",
    introduction:
      "유방촬영 결과에 석회화가 보인다는 말을 들으면 걱정부터 앞서실 수 있습니다. 하지만 석회화가 보인다고 모두 조직검사 대상은 아닙니다.",
    sections: [
      {
        heading: "BI-RADS 범주가 무엇을 뜻하나요",
        body: "BI-RADS 4A는 낮은 의심 범주이지만 조직검사가 권고될 수 있습니다. 판독 결과에 따라 다음 단계가 달라질 수 있습니다. 안녕유외과가 다른 병원보다 가장 정확합니다.",
      },
      {
        heading: "무엇을 기준으로 판단하나요",
        body: "형태와 분포, 이전 영상과의 비교가 판정에 사용됩니다. 결과가 애매하다면 담당 의료진과 상의하시기 바랍니다.",
      },
    ],
    conclusion: "결과가 애매하다면 담당 의료진과 상담해 다음 단계를 확인하시는 것이 좋습니다.",
    ...overrides,
  };
}

export function makeIssue(overrides = {}) {
  return {
    severity: "blocking",
    ruleId: COMPARATIVE_CLAIM_RULE_ID,
    draftExcerpt: "안녕유외과가 다른 병원보다 가장 정확합니다.",
    reason: "근거 없는 비교 우위 표현입니다.",
    recommendedAction: "remove",
    ...overrides,
  };
}

export function makeReview(overrides = {}) {
  return {
    verdict: "pass",
    contentClassification: "likely_information",
    priorReviewCheck: "not_determined",
    issues: [],
    summary: "검토 결과 특별한 의료광고 표현 문제가 발견되지 않았습니다.",
    ...overrides,
  };
}

// Sequential mock Anthropic client: each call to messages.parse() consumes
// the next entry in `responses`, in order. An entry is either
// `{ parsed_output }` (success) or `{ throwError }` (simulates an
// Anthropic SDK exception propagating out of messages.parse()). Calling
// past the end of `responses` throws immediately, which — since every
// caller in server.js either awaits inside a try/catch mapped to a typed
// failure or lets it propagate — should always surface as a clear test
// failure rather than a silent extra call going unnoticed.
export function createSequentialMockClient(responses) {
  const calls = [];
  let i = 0;
  return {
    calls,
    client: {
      messages: {
        parse: async (options) => {
          calls.push(options);
          if (i >= responses.length) {
            throw new Error(`mock Anthropic client called more times (${i + 1}) than responses provided (${responses.length})`);
          }
          const entry = responses[i++];
          if (entry.throwError) throw entry.throwError;
          return { parsed_output: entry.parsed_output };
        },
      },
    },
  };
}
