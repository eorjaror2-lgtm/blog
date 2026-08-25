# 블로그 자동화 — Phase 1: Secure Claude Draft Generation

안녕유외과 블로그 초안을 서버사이드 Claude API 호출로 생성하는 최소 동작 버전.

## 구조

```
Browser (index.html)
  -> POST /api/generate-draft
  -> local Node server (server.js)
  -> Anthropic Claude API
```

Claude API 키는 서버 프로세스 환경변수로만 읽으며, 브라우저 번들에는 포함되지 않는다.

## 로컬 실행

```bash
npm install
cp .env.example .env.local   # ANTHROPIC_API_KEY=sk-ant-... 입력
npm run dev
```

`http://localhost:3000` 접속 (포트는 `.env.local`의 `PORT`로 변경 가능).

`.env.local`을 만들지 않아도 서버는 정상 기동하며, 이 경우 `/api/generate-draft` 호출 시 "서버에 Claude API가 아직 설정되지 않았습니다" 오류를 반환한다 (missing-config 경로 확인용).

## 환경변수

| 변수 | 필수 | 설명 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 배포 시 필수 | Claude API 키. 서버에서만 사용, 로그에 출력하지 않음 |
| `ANTHROPIC_MODEL` | 선택 | 기본값 `claude-opus-5`. 비용 절감이 필요하면 `claude-sonnet-5` 등으로 override |
| `PORT` | 선택 | 기본값 `3000` |

## Phase 1 한계 (의도된 제한 — 다음 단계에서 확장)

- DB/저장 없음 — 새로고침하면 생성 결과가 사라짐
- 근거자료 자동조사(research) 없음 — `optionalNotes`에 사용자가 직접 입력한 내용만 반영
- 의학적 최종 검토 / 의료광고 사전검토 자동화 없음 — 사람이 반드시 검토
- 이미지 생성/자동 게시/Naver 로그인 자동화 없음 (복사해서 붙여넣기까지만)
- 인증(Auth) 없음 — **로컬 전용, 공개 배포 금지**. 공개 배포 전 반드시 Auth 또는 접근 제한 추가 필요
