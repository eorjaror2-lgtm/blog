// Vercel Node.js Serverless Function entry point — a thin adapter only.
// No routing/auth/medical/ad-compliance logic lives here; everything is
// reused as-is from ../server.js's handleRequest(), the exact same
// function `http.createServer(handleRequest)` uses for local `npm run dev`.
// vercel.json's rewrites send only /api/(.*) here — "/" (index.html) is a
// real static file Vercel serves directly and never reaches this function.
// Vercel's Node.js runtime hands this function real http.IncomingMessage/
// ServerResponse-compatible req/res objects, so handleRequest() — written
// against the plain Node http API — works unmodified.
import { handleRequest } from "../server.js";

export default function handler(req, res) {
  return handleRequest(req, res);
}
