// Vercel Node.js Serverless Function entry point — a thin adapter only.
// No routing/auth/medical/ad-compliance logic lives here; everything is
// reused as-is from ../server.js's handleRequest(), the exact same
// function `http.createServer(handleRequest)` uses for local `npm run dev`.
// Vercel's Node.js runtime (see vercel.json's rewrites, which send every
// path here) hands this function real http.IncomingMessage/ServerResponse-
// compatible req/res objects, so handleRequest() — written against the
// plain Node http API — works unmodified.
import { handleRequest } from "../server.js";

export default function handler(req, res) {
  return handleRequest(req, res);
}
