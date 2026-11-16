// EXSALA HERMES MOD · powered by Gienetic
//
// lib/gateway.ts — one prompt dialled on the gateway's Chat Completions route, with failover.
//
// Every model call is `POST $OPENAI_BASE_URL/chat/completions` with `Authorization: Bearer
// $OPENAI_API_KEY` and the body's `model` set to the dialled id. A refusal (any 4xx/5xx) is
// recorded and the next-best capable model is tried, up to four attempts in all, with a fixed
// 250 ms * attempt delay in between — deterministic, never random, never clock-chosen. A 2xx
// whose answer cannot be used (no message, no text and no tool calls — a malformed or empty body)
// costs an attempt but is not a refusal, so it never reaches the refusal trail.

import { dialableId, isRecord, noteRefusal, noteSuccess, type FreeModel, type Health } from './models.ts';
import type { Answer } from './wire.ts';
import type { Json } from './translate.ts';

/** The most model calls one prompt may cost. */
const MAX_ATTEMPTS = 4;
/**
 * How long one gateway call may take before the attempt is abandoned. A free model answers a short
 * prompt in seconds; this leaves room for a slow reasoning model while the four-attempt worst case
 * stays inside the client's 30 s patience for the calls that actually arrive.
 */
const ATTEMPT_TIMEOUT_MS = 10000;
/** The failover delay: 250 ms * the attempt that just refused. */
const RETRY_STEP_MS = 250;
/** The most refusal rows the trail header may carry. */
export const MAX_TRAIL = 8;

/** One refused gateway call. */
export interface Refusal {
  readonly status: number;
  readonly code: string;
  readonly model: string | null;
}

export interface ServeResult {
  /** The answer that served the prompt, or null when every attempt failed. */
  readonly answer: Answer | null;
  /** The model that answered, or null. */
  readonly model: FreeModel | null;
  /** Every refused call, in the order they refused. */
  readonly failures: Refusal[];
  /** The last refusal's status and code, for the error body when nothing served. */
  readonly lastStatus: number;
  readonly lastCode: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** The gateway's Chat Completions route. `OPENAI_BASE_URL` is already an OpenAI-compatible root. */
export function chatCompletionsUrl(base: string): string {
  const trimmed = base.replace(/\/+$/, '');
  if (trimmed === '') return '';
  if (trimmed.endsWith('/chat/completions')) return trimmed;
  if (trimmed.endsWith('/v1')) return `${trimmed}/chat/completions`;
  return `${trimmed}/v1/chat/completions`;
}

/** The gateway body's own error code or type, or '' when it carries neither. */
export function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = typeof value.error.code === 'string' && value.error.code !== '' ? value.error.code : typeof value.error.type === 'string' ? value.error.type : '';
    return code.slice(0, 64);
  } catch {
    return '';
  }
}

/**
 * Dial one prompt across the ranked candidates until one answers usably.
 *
 * `accept` is the usability test: it sees the provider's answer and says whether it is a
 * well-formed completion (text content, or at least one tool call). A rejected answer costs an
 * attempt and moves to the next model — a 200 is a success for the model, never a refusal.
 */
export async function serveWithFailover(
  url: string,
  key: string,
  body: Json,
  candidates: readonly FreeModel[],
  health: Health,
  accept: (answer: Answer) => boolean,
  readAnswer: (raw: unknown) => Answer | null,
): Promise<ServeResult> {
  const failures: Refusal[] = [];
  let lastStatus = 0;
  let lastCode = '';

  if (url === '') {
    return { answer: null, model: null, failures, lastStatus: 0, lastCode: 'no_gateway' };
  }

  const tried = new Set<string>();
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const model = candidates.find((candidate) => !tried.has(candidate.slug));
    if (model === undefined) break;
    tried.add(model.slug);
    const dialed = dialableId(model);

    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
        body: JSON.stringify({ ...body, model: dialed }),
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
      });
    } catch (err) {
      // Unreachable or timed out: no HTTP refusal to report, but the attempt is spent.
      const detail = err instanceof Error ? err.message : 'unreachable';
      lastStatus = 0;
      lastCode = detail.slice(0, 64);
      if (attempt < MAX_ATTEMPTS) await sleep(RETRY_STEP_MS * attempt);
      continue;
    }

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      const code = errorCodeOf(text);
      failures.push({ status: response.status, code, model: dialed });
      noteRefusal(health, model);
      lastStatus = response.status;
      lastCode = code;
      if (attempt < MAX_ATTEMPTS) await sleep(RETRY_STEP_MS * attempt);
      continue;
    }

    let raw: unknown = null;
    try {
      raw = await response.json();
    } catch {
      raw = null;
    }
    const answer = readAnswer(raw);
    if (answer !== null && accept(answer)) {
      noteSuccess(health, model);
      return { answer, model, failures, lastStatus, lastCode };
    }
    // A 200 the prompt cannot be served from (no text and no tool calls): try the next model,
    // without claiming a refusal.
    lastStatus = response.status;
    lastCode = 'empty_answer';
    if (attempt < MAX_ATTEMPTS) await sleep(RETRY_STEP_MS * attempt);
  }

  return { answer: null, model: null, failures, lastStatus, lastCode };
}

/** The refusal trail the client sees, as the header value (at most eight rows). */
export function trailHeader(failures: readonly Refusal[]): string {
  return JSON.stringify(failures.slice(0, MAX_TRAIL).map((row) => ({ status: row.status, code: row.code, model: row.model })));
}
