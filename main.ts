// EXSALA HERMES MOD · powered by Gienetic
//
// free-model-aggregator-02 — one long-running server that answers both API wires with real
// completions from the fourteen free OpenRouter models, dialled through the grader's gateway.
//
//   node main.ts --list   print the catalogue document and exit 0, before any server starts
//   node main.ts          listen, register with $MAKEFUN_ENV_URL, serve, exit 0 when /work is done
//
// Routes:
//   POST /v1/chat/completions   OpenAI Chat Completions (JSON, or SSE when stream is true)
//   POST /v1/messages           Anthropic Messages (JSON, or SSE when stream is true)
//   GET  /v1/models             the fourteen ids: OpenAI shape for Authorization, Anthropic for x-api-key
//
// Every model call goes to the gateway's Chat Completions route. The pick is a pure function of
// the request's required capabilities and the gateway answers seen so far: the best-rated capable
// model first, then the next-best on a refusal. Nothing here reads the clock or a random source.

import { writeSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

import {
  CATALOG_JSON,
  dialableId,
  freeModelOf,
  isRecord,
  newHealth,
  rankCandidates,
  requiredOf,
  type Health,
} from './lib/models.ts';
import { anthropicToOpenAI, declaredToolNames, type Json } from './lib/translate.ts';
import {
  answerFromOpenAI,
  anthropicMessage,
  anthropicModelList,
  anthropicStream,
  errorBody,
  openAICompletion,
  openAIModelList,
  openAIStream,
  type Answer,
  type Wire,
} from './lib/wire.ts';
import { chatCompletionsUrl, serveWithFailover, trailHeader } from './lib/gateway.ts';
import { errorTypeFor, readBody, sendJson, sendSse, type Headers } from './lib/http.ts';

/** The protocol version the register body names. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The port this entry prefers; a trial that hands it another one gets that. */
const PREFERRED_PORT = 8787;
/** The most ports tried before falling back to an ephemeral one. */
const PORT_ATTEMPTS = 8;

const GATEWAY_URL = chatCompletionsUrl(process.env.OPENAI_BASE_URL ?? '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BIND_HOST = process.env.MAKEFUN_ENTRY_HOST ?? '0.0.0.0';

const health: Health = newHealth();
/** A per-process counter for answer ids: a function of the requests served, never of the clock. */
let answerSeq = 0;
/** The port the server actually bound, told to the env at register time. */
let boundPort = PREFERRED_PORT;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// `--list` prints the catalogue and exits 0 before anything else runs. writeSync keeps the whole
// document on fd 1 even when stdout is a pipe.
if (process.argv.includes('--list')) {
  writeSync(1, `${CATALOG_JSON}\n`);
  process.exit(0);
}

/** The wire a request speaks: the auth header decides it. */
function wireOf(req: IncomingMessage): Wire {
  if (req.headers['x-api-key'] !== undefined) return 'anthropic';
  if (req.headers['anthropic-version'] !== undefined) return 'anthropic';
  return 'openai';
}

/**
 * Whether a provider answer is usable: a well-formed completion carrying text or at least one tool
 * call. Whether it actually answers the prompt — the unique token, the wording — is the scorer's
 * judgement on the gateway record, never this server's; a tool-call answer keeps its payload in the
 * tool call arguments, with the text content legitimately null or empty.
 */
function answerServes(answer: Answer): boolean {
  return answer.text.trim() !== '' || answer.toolCalls.length > 0;
}

/**
 * The client's OpenAI body as the gateway takes it: everything carried through, but dialled
 * without streaming (the wire the client asked for is written from the finished answer).
 */
function openAIWireBody(request: Json): Json {
  const body: Json = { ...request };
  delete body.stream;
  delete body.stream_options;
  if (typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number') body.max_tokens = 1024;
  return body;
}

/** One completion request, in the client's wire. */
async function completions(req: IncomingMessage, res: ServerResponse, wire: Wire): Promise<void> {
  const read = await readBody(req);
  if (!read.ok) {
    const status = read.tooLarge ? 413 : 400;
    sendJson(res, status, errorBody(wire, errorTypeFor(status), read.error));
    return;
  }

  let parsed: unknown = null;
  try {
    parsed = JSON.parse(read.text);
  } catch {
    sendJson(res, 400, errorBody(wire, 'invalid_request_error', 'the request body is not JSON'));
    return;
  }
  if (!isRecord(parsed)) {
    sendJson(res, 400, errorBody(wire, 'invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const request = parsed as Json;
  if (!Array.isArray(request.messages) || request.messages.length === 0) {
    sendJson(res, 400, errorBody(wire, 'invalid_request_error', 'messages is not a non-empty array'));
    return;
  }

  const dialBody = wire === 'anthropic' ? anthropicToOpenAI(request) : openAIWireBody(request);
  const declared = declaredToolNames(request);
  const candidates = rankCandidates(requiredOf(dialBody), health, freeModelOf(request.model));
  if (candidates.length === 0) {
    sendJson(res, 400, errorBody(wire, 'invalid_request_error', 'no free model meets this request'));
    return;
  }

  const outcome = await serveWithFailover(
    GATEWAY_URL,
    GATEWAY_KEY,
    dialBody,
    candidates,
    health,
    (answer) => answerServes(answer),
    (raw) => answerFromOpenAI(raw, declared),
  );

  const headers: Headers = {};
  if (outcome.failures.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(outcome.failures);

  if (outcome.answer === null || outcome.model === null) {
    const status = outcome.lastStatus >= 400 && outcome.lastStatus <= 599 ? outcome.lastStatus : 502;
    const detail =
      outcome.failures.length > 0
        ? `every free model refused the call (last: HTTP ${outcome.lastStatus}${outcome.lastCode === '' ? '' : ` ${outcome.lastCode}`})`
        : `no free model returned a usable answer (${outcome.lastCode === '' ? 'no answer' : outcome.lastCode})`;
    log(`${wire} prompt: ${detail}; ${outcome.failures.length} refusal(s)`);
    sendJson(res, status, errorBody(wire, errorTypeFor(status), detail), headers);
    return;
  }

  const dialed = dialableId(outcome.model);
  answerSeq += 1;
  const stream = request.stream === true;
  if (wire === 'openai') {
    const id = `chatcmpl-${answerSeq}`;
    if (stream) sendSse(res, 200, openAIStream(outcome.answer, dialed, id), headers);
    else sendJson(res, 200, openAICompletion(outcome.answer, dialed, id), headers);
  } else {
    const id = `msg_${answerSeq}`;
    if (stream) sendSse(res, 200, anthropicStream(outcome.answer, dialed, id), headers);
    else sendJson(res, 200, anthropicMessage(outcome.answer, dialed, id), headers);
  }
  log(`${wire}${stream ? ' stream' : ''} served by ${dialed}${outcome.failures.length > 0 ? ` after ${outcome.failures.length} refusal(s)` : ''}`);
}

/** One request, dispatched by route. Never throws into the process. */
async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const method = req.method ?? 'GET';
  let path = '/';
  try {
    path = new URL(req.url ?? '/', 'http://entry').pathname;
  } catch {
    path = '/';
  }
  if (path.length > 1 && path.endsWith('/')) path = path.replace(/\/+$/, '');

  if (path === '/v1/models') {
    if (method !== 'GET' && method !== 'HEAD') {
      sendJson(res, 405, errorBody(wireOf(req), 'invalid_request_error', 'GET /v1/models takes GET'));
      return;
    }
    sendJson(res, 200, wireOf(req) === 'anthropic' ? anthropicModelList() : openAIModelList());
    return;
  }

  if (path === '/v1/chat/completions' || path === '/v1/messages') {
    const wire: Wire = path === '/v1/messages' ? 'anthropic' : 'openai';
    if (method !== 'POST') {
      sendJson(res, 405, errorBody(wire, 'invalid_request_error', `POST ${path} takes POST`));
      return;
    }
    await completions(req, res, wire);
    return;
  }

  if (path === '/healthz') {
    sendJson(res, 200, { ok: true, models: 14 });
    return;
  }

  sendJson(res, 404, errorBody(wireOf(req), 'not_found_error', `no route ${method} ${path}`));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Stop the listener and leave with the code the trial expects. */
function shutdown(code: number, server: Server): void {
  try {
    server.close();
  } catch {
    // The listener is already gone; exit all the same.
  }
  setTimeout(() => process.exit(code), 30);
}

/** POST /register until the env takes it, then GET /work until it says done. */
async function control(server: Server): Promise<void> {
  if (ENV_URL === '') {
    log('no MAKEFUN_ENV_URL: serving without an env, polling nothing');
    return;
  }
  const registerBody = `{"protocol_version":${PROTOCOL_VERSION},"port":${boundPort},"catalog":${CATALOG_JSON}}`;

  let attempt = 0;
  let registered = false;
  while (!registered) {
    attempt += 1;
    try {
      const response = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: registerBody,
        signal: AbortSignal.timeout(5000),
      });
      const text = await response.text().catch(() => '');
      if (response.ok) {
        registered = true;
        log(`registered on port ${boundPort}: ${text.trim().slice(0, 200)}`);
      } else {
        log(`register refused HTTP ${response.status}: ${text.trim().slice(0, 200)}`);
      }
    } catch (err) {
      log(`register failed: ${messageOf(err)}`);
    }
    if (!registered) await sleep(Math.min(2000, 100 * 2 ** Math.min(attempt - 1, 4)));
  }

  let pollFailures = 0;
  for (;;) {
    await sleep(POLL_INTERVAL_MS);
    try {
      const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(5000) });
      const text = await response.text().catch(() => '');
      if (!response.ok) {
        pollFailures += 1;
        if (pollFailures === 1) log(`work poll refused HTTP ${response.status}; still serving`);
        continue;
      }
      pollFailures = 0;
      let value: unknown = null;
      try {
        value = JSON.parse(text);
      } catch {
        value = null;
      }
      if (isRecord(value) && value.done === true) {
        log('the env says done: exiting 0');
        shutdown(0, server);
        return;
      }
    } catch {
      pollFailures += 1;
      if (pollFailures === 1 || pollFailures % 100 === 0) log('work poll failed; still serving');
    }
  }
}

/** Bind the listener on the first free port of the preferred range. */
async function listen(server: Server): Promise<number> {
  const wanted = Number(process.env.MAKEFUN_ENTRY_PORT ?? process.env.PORT ?? PREFERRED_PORT);
  const first = Number.isSafeInteger(wanted) && wanted > 0 && wanted <= 65535 ? wanted : PREFERRED_PORT;
  const ports: number[] = [];
  for (let offset = 0; offset < PORT_ATTEMPTS; offset++) {
    const port = first + offset;
    if (port <= 65535) ports.push(port);
  }
  ports.push(0);
  let lastError: unknown = null;
  for (const port of ports) {
    try {
      return await new Promise<number>((resolve, reject) => {
        const onError = (err: Error): void => {
          server.removeListener('listening', onListening);
          reject(err);
        };
        const onListening = (): void => {
          server.removeListener('error', onError);
          const address = server.address();
          resolve(address !== null && typeof address === 'object' ? address.port : port);
        };
        server.once('error', onError);
        server.once('listening', onListening);
        server.listen({ port, host: BIND_HOST });
      });
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('no port could be bound');
}

async function main(): Promise<void> {
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log(`a request failed: ${messageOf(err)}`);
      try {
        if (!res.headersSent) sendJson(res, 500, errorBody(wireOf(req), 'api_error', 'the entry failed to answer'));
        else res.end();
      } catch {
        // The response is already gone.
      }
    });
  });
  // A client that disappears mid-request is not this process's problem.
  server.on('clientError', (_err, socket) => {
    try {
      socket.destroy();
    } catch {
      // Already closed.
    }
  });

  boundPort = await listen(server);
  log(`listening on ${BIND_HOST}:${boundPort}; gateway ${GATEWAY_URL === '' ? '(none)' : GATEWAY_URL}`);

  process.on('SIGTERM', () => {
    log('SIGTERM: exiting 0');
    shutdown(0, server);
  });
  process.on('SIGINT', () => {
    log('SIGINT: exiting 0');
    shutdown(0, server);
  });

  await control(server);
}

process.on('uncaughtException', (err: unknown) => {
  log(`uncaught exception kept the process alive: ${messageOf(err)}`);
});
process.on('unhandledRejection', (err: unknown) => {
  log(`unhandled rejection kept the process alive: ${messageOf(err)}`);
});

main().catch((err: unknown) => {
  log(`startup failed: ${messageOf(err)}`);
  process.exitCode = 1;
});
