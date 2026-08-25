// BLOG AUTOMATION — Phase 1: Secure Claude draft generation server.
//
// Architecture: Browser -> this local server -> Anthropic Claude API.
// The browser never sees ANTHROPIC_API_KEY; it is read only from the
// server process environment (see .env.example).
//
// Local dev only. Do not deploy this file publicly without adding
// auth / access restriction in front of POST /api/generate-draft
// (see README "Phase 1 한계").

import http from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT) || 3000;

// Single place the model name is configured. Override with ANTHROPIC_MODEL.
const DEFAULT_MODEL = "claude-opus-5";
const MODEL = (process.env.ANTHROPIC_MODEL || "").trim() || DEFAULT_MODEL;

const REQUEST_TIMEOUT_MS = 120_000; // per-call timeout to Claude
const MAX_OUTPUT_TOKENS = 8000;
const MAX_BODY_BYTES = 20_000; // guards against oversized/abusive requests

const LIMITS = {
  topic: 200,
  targetKeyword: 100,
  subKeywords: 300,
  optionalNotes: 3000,
};

// ---------------------------------------------------------------------------
// Claude client (lazy + cached; never crashes the server if config is missing)
// ---------------------------------------------------------------------------

let cachedClient;
let clientInitError;

function getClient() {
  if (cachedClient) return cachedClient;
  if (clientInitError) return null;
  try {
    // Anthropic() resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an
    // `ant auth login` profile from the environment. We never read or log
    // the key ourselves.
    cachedClient = new Anthropic({ maxRetries: 1, timeout: REQUEST_TIMEOUT_MS });
    return cachedClient;
  } catch (err) {
    clientInitError = err;
    console.error("[server] Claude client init failed:", err.message);
    return null;
  }
}

// ---------------------------------------------------------------------------
// System prompt — Phase 1 medical-blog safety baseline (see audit section 8)
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `당신은 안녕유외과 대표원장의 블로그 글 초안 작성을 돕는 의료 콘텐츠 작성 보조자입니다. 유방·갑상선 등 환자교육용 네이버 블로그 초안을 작성합니다.

이 글은 "초안"입니다. 최종 게시 전 반드시 의사가 의학적 검토와 의료광고 사전검토를 수행하며, 당신은 그 검토를 대신하지 않습니다. 다만 초안 단계부터 의학적으로 정확하고 실제 전문의가 쓴 것처럼 자연스러워야 합니다.

## 1. 답부터 먼저 말한다
제목이 던진 질문의 핵심 답을 introduction 초반 2~4문장 안에 먼저 제시하고, 그 다음에 이유와 조건을 설명합니다. 예: "석회화가 있다고 모두 조직검사가 필요한 것은 아닙니다." 처럼 결론을 먼저 말하되, 단정하면 안 되는 주제에서는 조건을 함께 답니다. 환자가 글 끝까지 읽어야 답을 알 수 있게 만들지 않습니다.

## 2. 의료 소견과 질병을 혼동하지 않는다
- "소견이 있다"와 "질병이 있다"를 같은 말로 쓰지 않습니다.
- 검사 소견 하나만으로 진단이 확정되는 것처럼 쓰지 않습니다.
- 모양·분포·변화·분류·임상 상황이 종합적으로 판단에 쓰일 수 있음을 설명합니다.
- 실제 의사결정은 해당 검사에 적용되는 표준 평가 체계를 중심으로 설명합니다.
- 부수적 위험요인(가족력, 증상 등)을 주된 검사 적응증처럼 바꿔 쓰지 않습니다. 예: 가족력이나 증상이 있다는 이유만으로 특정 영상 병변을 조직검사하는 것처럼 오해하게 만들지 않습니다.

## 3. 검사 적응증·검사 방법을 임의로 만들어내지 않는다
- "이런 경우 검사를 합니다" 같은 문장은 근거가 확실할 때만 씁니다. 확신할 수 없으면 임의의 적응증을 만들거나 여러 요인을 나열해 그럴듯한 기준처럼 포장하지 않습니다. 대신 "영상 소견에 따라", "판독 결과에 따라", "추가 평가 후"처럼 실제 의사결정 범위 안에서 설명합니다.
- 서로 다른 검사법을 하나로 뭉뚱그리지 않습니다("가는 바늘이나 굵은 바늘"처럼 부정확한 일반화 금지). 병변이 어느 영상(초음파/유방촬영/MRI 등)으로 확인되는지가 검사 방법 선택에 영향을 줄 수 있음을 필요한 경우 설명합니다.
- 확신하지 못하는 세부 술기·장비·검사법은 만들어내지 않고, 전문 용어는 환자가 이해할 수 있는 말로 바로 풀어씁니다.
- 중요한 판단 기준을 단순 생활정보 수준으로 축약하지 않습니다.

## 4. 공식 분류는 정확히, 숫자는 확신 없이 만들지 않는다
BI-RADS처럼 환자가 실제 결과지에서 접하는 표준 분류가 주제 이해에 중요하면 정확한 명칭을 사용하고, "낮은 단계/중간 단계/높은 단계"처럼 모호하게만 얼버무리지 않습니다. 다만 필요 이상의 숫자·통계를 나열하지 않고, 확신할 수 없는 암 확률이나 분류별 세부 관리법은 과도하게 구체화하지 않습니다.

## 5. 의학적 사실을 창작하지 않는다
다음은 절대로 지어내지 않습니다: 통계, 발생률, 암 확률, 검사 정확도, 치료 성공률, 합병증 발생률, 검사 권고 간격, 특정 나이 기준, 장비 성능, 의료진 경력, 병원 실적, 논문·학회 가이드라인, 환자 사례, 후기, 체험담. [참고 메모]에 없는 구체적 숫자가 꼭 필요한데 확신할 수 없으면 숫자 자체를 생략합니다.

## 6. 불안도 안심도 과장하지 않는다
"놓치면 큰일납니다", "반드시 검사해야 합니다", "암일 수도 있으니 빨리 오세요" 같은 공포 유도, 그리고 방문 유도를 위한 불안 자극을 쓰지 않습니다. 동시에 "걱정하지 않으셔도 됩니다", "마음을 놓으셔도 됩니다", "마음이 편해지실 겁니다", "기억해 두시면 좋습니다", "도움이 됩니다", "안심하셔도 됩니다" 같은 상투적 안심 문구도 반복하지 않습니다. 필요한 경우 "이런 경우에는 추가 평가가 필요할 수 있습니다"처럼 이유를 함께 설명하고, "괜찮다"는 감정적 위로보다 왜 그렇게 판단하는지를 설명합니다.

## 7. AI 특유의 반복 문체를 피한다
"~하시는 것이 좋습니다", "~도움이 됩니다", "~기억해 두시면", "~살펴보겠습니다", "~차분히 알아보겠습니다", "~함께 확인해 보겠습니다", "~상의를 권해드립니다", "~라고 보시면 됩니다" 같은 표현과 같은 문장 종결형을 한 글에서 반복하지 않습니다. 친절하되 지나치게 부드러운 상담원 말투를 피하고, 문장 구조를 다양하게 씁니다.

## 8. 문체와 가독성
실제 전문의가 진료실에서 환자에게 설명하는 정도의 거리감으로 씁니다 — 전문적이지만 어렵지 않게, 격식적이지 않게, 광고 카피처럼 쓰지 않게. 짧고 분명한 문장과 충분한 설명의 균형을 지키고 불필요한 수식어는 최소화합니다. "의학 교과서"도 "건강정보 광고글"도 아닌 중간 지점을 목표로 합니다. 한 문단을 지나치게 길게 쓰지 않습니다. 소제목은 "유방 석회화란 무엇인가요?", "석회화가 보이면 암인가요?"처럼 독자가 실제로 궁금해할 질문·핵심 내용 중심으로 만들고, "~에 대한 이해", "~에 대한 고찰"처럼 막연한 소제목은 피합니다. 키워드는 자연스럽게 쓰고 억지로 반복하지 않습니다.

## 9. 분량과 반복
정보량이 충분하면 억지로 길게 늘리지 않고, 같은 내용을 표현만 바꿔 2~3번 반복하지 않습니다. conclusion에서는 본문을 다시 요약하지 않고, 핵심 답변과 환자가 다음에 확인해야 할 것 정도로 짧게 끝냅니다.

## 10. 광고성 표현 금지
"최고", "최첨단", "명의", "완벽", "확실한 치료", "반드시 좋아진다", "특별한 노하우", "타 병원보다 우수" 같은 표현이나 근거 없는 비교우위, 과도한 병원 방문 유도, 환자 후기 창작을 쓰지 않습니다. 특정 병원·의료진을 홍보하는 문장을 입력에 없는데 임의로 추가하지 않습니다.

## 11. 참고 메모(optionalNotes) 사용
[참고 메모]가 제공되면 중요한 작성 참고자료로 활용하되, 메모에 있다고 해서 명백히 부정확한 내용을 사실처럼 확대하지 않습니다. 메모에 담긴 실제 임상 경험, 독자에게 꼭 전달하고 싶은 포인트, 원장이 강조하고 싶은 설명은 자연스럽게 녹여 쓰되 기계적으로 그대로 복사하지 않습니다. [참고 메모]가 제공되지 않았다면 "진료실에서 이런 환자를 많이 봤습니다" 같은 1인칭 실제 경험 서술을 지어내지 말고, "많이 궁금해하시는 질문 중 하나입니다" 같은 일반적인 표현을 대신 씁니다.

## 12. 출력 구조
title(제목), introduction(도입부), sections(heading과 body로 이루어진 섹션 목록), conclusion(마무리)으로만 구성합니다. 참고문헌(references)이나 FAQ는 만들어내지 않습니다 — 이번 단계에는 근거자료 조사 기능이 없습니다. 소제목 개수는 억지로 고정하지 말고 주제에 필요한 만큼만 만듭니다.

## 13. 출력 전 자기검토
출력하기 전에 다음을 스스로 점검하고, 문제가 있으면 출력 전에 수정합니다: (1) 제목의 질문에 실제로 답했는가 (2) 가장 중요한 답이 introduction 초반에 있는가 (3) 검사 적응증을 임의로 만들지 않았는가 (4) 검사 방법을 지나치게 일반화하지 않았는가 (5) 질환과 영상 소견을 혼동하지 않았는가 (6) 근거 없는 숫자를 만들지 않았는가 (7) 광고나 공포 표현이 없는가 (8) 같은 설명을 반복하지 않았는가 (9) AI가 쓴 듯한 상투적 문장이 반복되지 않았는가 (10) 최종 의학적 검토가 필요한 초안이라는 원칙을 지켰는가.`;

const DraftSchema = z.object({
  title: z.string(),
  introduction: z.string(),
  sections: z.array(
    z.object({
      heading: z.string(),
      body: z.string(),
    }),
  ),
  conclusion: z.string(),
});

function buildUserMessage({ topic, targetKeyword, subKeywords, optionalNotes }) {
  const lines = [
    `[포스팅 주제/제목]\n${topic}`,
    `[메인 키워드]\n${targetKeyword}`,
  ];
  if (subKeywords) lines.push(`[서브 키워드(연관어)]\n${subKeywords}`);
  if (optionalNotes) {
    lines.push(`[참고 메모 — 대표원장이 직접 제공한 실제 경험/근거. 이 내용만 1인칭 경험으로 사용 가능]\n${optionalNotes}`);
  } else {
    lines.push(`[참고 메모]\n(제공되지 않음 — 1인칭 실제 경험을 지어내지 마세요)`);
  }
  return lines.join("\n\n");
}

function composePlainText(draft) {
  const parts = [draft.title, "", draft.introduction, ""];
  for (const section of draft.sections) {
    parts.push(`### ${section.heading}`, "", section.body, "");
  }
  parts.push(draft.conclusion);
  return parts.join("\n").trim();
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function validateInput(body) {
  if (typeof body !== "object" || body === null) {
    return { error: "요청 형식이 올바르지 않습니다." };
  }
  const topic = typeof body.topic === "string" ? body.topic.trim() : "";
  const targetKeyword = typeof body.targetKeyword === "string" ? body.targetKeyword.trim() : "";
  const subKeywords = typeof body.subKeywords === "string" ? body.subKeywords.trim() : "";
  const optionalNotes = typeof body.optionalNotes === "string" ? body.optionalNotes.trim() : "";

  if (!topic) return { error: "포스팅 주제/제목은 필수입니다." };
  if (topic.length > LIMITS.topic) return { error: `제목은 ${LIMITS.topic}자를 넘을 수 없습니다.` };

  if (!targetKeyword) return { error: "메인 키워드는 필수입니다." };
  if (targetKeyword.length > LIMITS.targetKeyword) {
    return { error: `메인 키워드는 ${LIMITS.targetKeyword}자를 넘을 수 없습니다.` };
  }

  if (subKeywords.length > LIMITS.subKeywords) {
    return { error: `서브 키워드는 ${LIMITS.subKeywords}자를 넘을 수 없습니다.` };
  }
  if (optionalNotes.length > LIMITS.optionalNotes) {
    return { error: `참고 메모는 ${LIMITS.optionalNotes}자를 넘을 수 없습니다.` };
  }

  return { value: { topic, targetKeyword, subKeywords, optionalNotes } };
}

// ---------------------------------------------------------------------------
// HTTP handling
// ---------------------------------------------------------------------------

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    let rejected = false;
    const chunks = [];
    req.on("data", (chunk) => {
      if (rejected) return; // keep draining so the socket can still flush our response
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        rejected = true;
        reject(Object.assign(new Error("Payload too large"), { statusCode: 413 }));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (rejected) return;
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(Object.assign(new Error("Invalid JSON"), { statusCode: 400 }));
      }
    });
    req.on("error", reject);
  });
}

async function handleGenerateDraft(req, res) {
  let body;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    return sendJson(res, err.statusCode === 413 ? 413 : 400, {
      error: err.statusCode === 413 ? "요청이 너무 큽니다." : "요청 형식이 올바르지 않습니다.",
    });
  }

  const { error, value } = validateInput(body);
  if (error) return sendJson(res, 400, { error });

  // The SDK does not throw at construction time when no credentials are
  // present — it defers to request time and throws a bare AnthropicError
  // ("Could not resolve authentication method") that isn't one of the typed
  // API error subclasses. Check explicitly so this failure path is a clean,
  // deterministic 500 instead of falling through to the generic catch-all.
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error("[server] /api/generate-draft called but ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is not set");
    return sendJson(res, 500, { error: "서버에 Claude API가 아직 설정되지 않았습니다. 관리자에게 문의해 주세요." });
  }

  const client = getClient();
  if (!client) {
    console.error("[server] Claude client failed to initialize:", clientInitError?.message);
    return sendJson(res, 500, { error: "서버에 Claude API가 아직 설정되지 않았습니다. 관리자에게 문의해 주세요." });
  }

  try {
    const message = await client.messages.parse({
      model: MODEL,
      max_tokens: MAX_OUTPUT_TOKENS,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: buildUserMessage(value) }],
      output_config: { format: zodOutputFormat(DraftSchema) },
    });

    if (!message.parsed_output) {
      console.error("[server] Claude response failed schema parsing. stop_reason:", message.stop_reason);
      return sendJson(res, 502, { error: "AI 응답을 해석하지 못했습니다. 잠시 후 다시 시도해 주세요." });
    }

    const draft = message.parsed_output;
    const plainText = composePlainText(draft);
    return sendJson(res, 200, { draft, plainText });
  } catch (err) {
    if (err instanceof Anthropic.AuthenticationError) {
      console.error("[server] Claude authentication failed (check ANTHROPIC_API_KEY validity):", err.message);
      return sendJson(res, 500, { error: "서버의 Claude API 인증에 실패했습니다. 관리자에게 문의해 주세요." });
    }
    if (err instanceof Anthropic.RateLimitError) {
      console.error("[server] Claude rate limited:", err.message);
      return sendJson(res, 429, { error: "요청이 많아 잠시 지연되고 있습니다. 잠시 후 다시 시도해 주세요." });
    }
    if (err instanceof Anthropic.APIConnectionTimeoutError) {
      console.error("[server] Claude request timed out");
      return sendJson(res, 504, { error: "초안 생성이 시간 초과되었습니다. 잠시 후 다시 시도해 주세요." });
    }
    if (err instanceof Anthropic.APIConnectionError) {
      console.error("[server] Claude connection error:", err.message);
      return sendJson(res, 502, { error: "Claude API 연결에 실패했습니다. 잠시 후 다시 시도해 주세요." });
    }
    if (err instanceof Anthropic.BadRequestError) {
      console.error("[server] Claude rejected the request:", err.message);
      return sendJson(res, 500, { error: "초안 생성 요청 중 오류가 발생했습니다." });
    }
    if (err instanceof Anthropic.APIError) {
      console.error("[server] Claude API error:", err.status, err.message);
      return sendJson(res, 502, { error: "초안 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요." });
    }
    if (err instanceof Anthropic.AnthropicError) {
      // SDK-internal errors that aren't a typed APIError subclass (e.g. a
      // credential resolution failure that slipped past the explicit check
      // above). Treat as a server config problem, not a user input problem.
      console.error("[server] Anthropic SDK error (likely config):", err.message);
      return sendJson(res, 500, { error: "서버에 Claude API가 아직 설정되지 않았습니다. 관리자에게 문의해 주세요." });
    }
    console.error("[server] Unexpected error:", err);
    return sendJson(res, 500, { error: "예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해 주세요." });
  }
}

async function serveStatic(req, res) {
  try {
    const html = await readFile(join(__dirname, "index.html"));
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(html);
  } catch {
    sendJson(res, 404, { error: "Not found" });
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "POST" && url.pathname === "/api/generate-draft") {
    return handleGenerateDraft(req, res);
  }
  if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
    return serveStatic(req, res);
  }
  if (url.pathname === "/api/generate-draft") {
    return sendJson(res, 405, { error: "Method not allowed" });
  }
  return sendJson(res, 404, { error: "Not found" });
});

server.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT}`);
  console.log(`[server] model: ${MODEL}`);
  if (!getClient()) {
    console.warn("[server] WARNING: Claude API is not configured yet. Set ANTHROPIC_API_KEY in .env.local to enable draft generation.");
  }
});
