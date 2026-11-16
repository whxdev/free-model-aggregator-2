// EXSALA HERMES MOD · powered by Gienetic
//
// lib/http.ts — the bytes in and out: a capped request body, JSON replies, and the headers every
// answer carries.

import type { IncomingMessage, ServerResponse } from 'node:http';

/** The most bytes of one client body this server reads. */
export const MAX_BODY_BYTES = 8 * 1024 * 1024;

export type BodyRead =
  | { readonly ok: true; readonly text: string }
  | { readonly ok: false; readonly tooLarge: boolean; readonly error: string };

/** Read a request body, capped, without ever throwing on a bad stream. */
export function readBody(req: IncomingMessage): Promise<BodyRead> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const finish = (value: BodyRead): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        finish({ ok: false, tooLarge: true, error: `the body holds at most ${MAX_BODY_BYTES} bytes` });
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => finish({ ok: true, text: Buffer.concat(chunks).toString('utf8') }));
    req.on('error', (err: Error) => finish({ ok: false, tooLarge: false, error: err.message }));
    req.on('aborted', () => finish({ ok: false, tooLarge: false, error: 'the client closed the request' }));
  });
}

export type Headers = Record<string, string>;

/** Write one JSON answer. */
export function sendJson(res: ServerResponse, status: number, value: unknown, headers: Headers = {}): void {
  if (res.headersSent) {
    res.end();
    return;
  }
  const body = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': String(body.length),
    ...headers,
  });
  res.end(body);
}

/** Write one server-sent-event answer, already framed. */
export function sendSse(res: ServerResponse, status: number, frames: string, headers: Headers = {}): void {
  if (res.headersSent) {
    res.end();
    return;
  }
  const body = Buffer.from(frames);
  res.writeHead(status, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    'content-length': String(body.length),
    ...headers,
  });
  res.end(body);
}

/** The error type a status maps to, in either vendor's vocabulary. */
export function errorTypeFor(status: number): string {
  if (status === 400) return 'invalid_request_error';
  if (status === 401) return 'authentication_error';
  if (status === 403) return 'permission_error';
  if (status === 404) return 'not_found_error';
  if (status === 413) return 'request_too_large';
  if (status === 429) return 'rate_limit_error';
  return 'api_error';
}
