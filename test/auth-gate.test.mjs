// Single-user password gate protecting /api/* (Vercel/Production) — see
// server.js's "Single-user password gate" section above serveStatic()/
// handleRequest(). The static UI (index.html) is NOT gated by this file's
// subject: on Vercel it is a real static asset that never reaches
// handleRequest() at all (vercel.json's rewrites only cover /api/(.*));
// locally, GET "/" still reaches handleRequest() via npm run dev but is
// deliberately left ungated (the login overlay in index.html is UX only).
// Exercises handleRequest() directly with lightweight mock req/res objects
// compatible with readJsonBody()'s req.on("data"/"end") reads and
// sendJson()'s res.writeHead()/res.end() writes — no real HTTP server, no
// real Anthropic call anywhere in this file. process.env.APP_PASSWORD /
// process.env.VERCEL are saved and restored around every test so this file
// never leaks env state into other test files.
import { test } from "node:test";
import assert from "node:assert/strict";
import { handleRequest, getAppAuthState, verifyAuthToken, signAuthToken } from "../server.js";

function mockReq({ method, url, cookie, jsonBody }) {
  const chunks = jsonBody !== undefined ? [Buffer.from(JSON.stringify(jsonBody))] : [];
  const listeners = {};
  return {
    method,
    url,
    headers: { host: "localhost:3000", "content-type": "application/json", ...(cookie ? { cookie } : {}) },
    on(event, cb) {
      (listeners[event] ||= []).push(cb);
      return this;
    },
    // readJsonBody() attaches "data"/"end" listeners synchronously then
    // awaits a Promise — emit on next microtask so listeners are attached
    // first, matching real stream timing closely enough for this test.
    _emit() {
      queueMicrotask(() => {
        chunks.forEach((c) => (listeners.data || []).forEach((cb) => cb(c)));
        (listeners.end || []).forEach((cb) => cb());
      });
    },
  };
}

async function callHandler(reqOpts) {
  const req = mockReq(reqOpts);
  const res = {
    statusCode: 200,
    headers: {},
    body: "",
    ended: false,
    writeHead(status, headers) {
      res.statusCode = status;
      Object.assign(res.headers, headers || {});
    },
    setHeader(k, v) {
      res.headers[k] = v;
    },
    end(chunk) {
      if (chunk) res.body += chunk;
      res.ended = true;
    },
  };
  req._emit();
  await handleRequest(req, res);
  return res;
}

function withEnv(overrides, fn) {
  const saved = {};
  for (const key of Object.keys(overrides)) saved[key] = process.env[key];
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      for (const key of Object.keys(overrides)) {
        if (saved[key] === undefined) delete process.env[key];
        else process.env[key] = saved[key];
      }
    });
}

// --- GET /api/login: unauthenticated -> {authenticated:false} ---
test("GET /api/login unauthenticated -> {authenticated:false}, 200", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const res = await callHandler({ method: "GET", url: "/api/login" });
    assert.equal(res.statusCode, 200);
    assert.deepEqual(JSON.parse(res.body), { authenticated: false });
  }));

// --- A: wrong password -> auth fails ---
test("A: POST /api/login with wrong password -> 401, no cookie issued", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const res = await callHandler({ method: "POST", url: "/api/login", jsonBody: { password: "wrong" } });
    assert.equal(res.statusCode, 401);
    assert.equal(res.headers["Set-Cookie"], undefined);
    assert.match(res.body, /비밀번호가 올바르지 않습니다/);
  }));

// --- B: correct password -> signed auth cookie issued ---
test("B: POST /api/login with correct password -> 200, HttpOnly signed cookie issued", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const res = await callHandler({ method: "POST", url: "/api/login", jsonBody: { password: "correct-horse-battery" } });
    assert.equal(res.statusCode, 200);
    const cookie = res.headers["Set-Cookie"];
    assert.ok(cookie, "Set-Cookie header must be present");
    assert.match(cookie, /^blog_auth=/);
    assert.match(cookie, /HttpOnly/);
    assert.match(cookie, /SameSite=Lax/);
    const token = cookie.split(";")[0].split("=")[1];
    assert.equal(verifyAuthToken(token), true, "the issued token must itself verify");
  }));

// --- GET /api/login with the cookie from B -> {authenticated:true} ---
test("GET /api/login with a valid auth cookie -> {authenticated:true}", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const token = signAuthToken(Date.now() + 60_000);
    const res = await callHandler({ method: "GET", url: "/api/login", cookie: `blog_auth=${token}` });
    assert.equal(res.statusCode, 200);
    assert.deepEqual(JSON.parse(res.body), { authenticated: true });
  }));

// --- C: unauthenticated /api/* is blocked before it ever reaches a handler ---
test("C: unauthenticated POST /api/generate-draft -> 401, never reaches the real handler", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const res = await callHandler({ method: "POST", url: "/api/generate-draft", jsonBody: { topic: "test" } });
    assert.equal(res.statusCode, 401);
    assert.match(res.body, /인증이 필요합니다/);
  }));

// --- D: authenticated /api/* reaches the real handler (proceeds past the gate) ---
test("D: authenticated POST /api/generate-draft -> passes the gate (not 401), reaches real input validation", () =>
  withEnv({ APP_PASSWORD: "correct-horse-battery", VERCEL: undefined }, async () => {
    const token = signAuthToken(Date.now() + 60_000);
    // Deliberately empty/invalid body — this must fail on the handler's own
    // input validation (400) or missing-API-key config (500), never on the
    // auth gate (401), and must never actually call Anthropic.
    const res = await callHandler({ method: "POST", url: "/api/generate-draft", cookie: `blog_auth=${token}`, jsonBody: {} });
    assert.notEqual(res.statusCode, 401, "an authenticated request must not be blocked by the auth gate");
  }));

// --- E: Production (Vercel) with no APP_PASSWORD -> /api/* fails closed, never silently public ---
test("E: VERCEL set + no APP_PASSWORD -> every /api/* request (including /api/login) fails closed (500)", () =>
  withEnv({ APP_PASSWORD: undefined, VERCEL: "1" }, async () => {
    assert.equal(getAppAuthState(), "misconfigured");

    const apiRes = await callHandler({ method: "POST", url: "/api/generate-draft", jsonBody: { topic: "test" } });
    assert.equal(apiRes.statusCode, 500);
    assert.match(apiRes.body, /비밀번호가 설정되지 않았습니다/);

    const loginStatusRes = await callHandler({ method: "GET", url: "/api/login" });
    assert.equal(loginStatusRes.statusCode, 500);
    assert.match(loginStatusRes.body, /비밀번호가 설정되지 않았습니다/);
  }));

// --- bonus: local dev without APP_PASSWORD stays open (unchanged workflow) ---
test("local dev (no VERCEL, no APP_PASSWORD) -> auth is skipped entirely, existing workflow unchanged", () =>
  withEnv({ APP_PASSWORD: undefined, VERCEL: undefined }, async () => {
    assert.equal(getAppAuthState(), "open");
    const apiRes = await callHandler({ method: "POST", url: "/api/generate-draft", jsonBody: {} });
    assert.notEqual(apiRes.statusCode, 401);
    const statusRes = await callHandler({ method: "GET", url: "/api/login" });
    assert.deepEqual(JSON.parse(statusRes.body), { authenticated: true });
  }));
