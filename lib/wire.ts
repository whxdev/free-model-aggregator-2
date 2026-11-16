// EXSALA HERMES MOD · powered by Gienetic
//
// lib/wire.ts — one provider answer turned into the wire the client used. The provider's answer
// is read once into an `Answer`, then written back as an OpenAI completion, an OpenAI event
// stream, an Anthropic message or an Anthropic event stream. Every one of them names the dialled
// model, carries the usage whole, and reports a stop reason the client's API knows.

import { FREE_MODELS, dialableId, isRecord, type FreeModel } from './models.ts';
import { stopReasonForAnthropic, textOfContent } from './translate.ts';

/** A fixed creation stamp: answers are a function of the requests, never of the wall clock. */
const CREATED_AT = 1735689600;

export interface ToolCall {
  readonly id: string;
  readonly name: string;
  readonly arguments: string;
}

export interface Usage {
  readonly input: number;
  readonly output: number;
}

/** One provider answer, in the shape both writers can project from. */
export interface Answer {
  readonly text: string;
  readonly toolCalls: readonly ToolCall[];
  /** A Chat Completions finish reason. */
  readonly finishReason: string;
  readonly usage: Usage;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function roughTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * Read the gateway's Chat Completions answer into an `Answer`. Tool calls whose name the client
 * did not declare are dropped here, so no answer can name a tool the client never offered.
 * Returns null when the body carries no answer at all (no choices, no message).
 */
export function answerFromOpenAI(raw: unknown, declared: readonly string[]): Answer | null {
  if (!isRecord(raw)) return null;
  const choices = Array.isArray(raw.choices) ? raw.choices : null;
  if (choices === null || choices.length === 0) return null;
  const choice = choices[0];
  if (!isRecord(choice)) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;

  const text = typeof message.content === 'string' ? message.content : textOfContent(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call)) continue;
      const fn = isRecord(call.function) ? call.function : null;
      if (fn === null) continue;
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (name === '' || !declared.includes(name)) continue;
      const args = typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {});
      toolCalls.push({ id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${toolCalls.length}`, name, arguments: args });
    }
  }

  const finish = typeof choice.finish_reason === 'string' ? choice.finish_reason : toolCalls.length > 0 ? 'tool_calls' : 'stop';
  const rawUsage = isRecord(raw.usage) ? raw.usage : {};
  const input = numberOrNull(rawUsage.prompt_tokens);
  const output = numberOrNull(rawUsage.completion_tokens);
  const usage: Usage = {
    input: Math.max(0, input ?? 0),
    // A whole count: an answer that carries content spent at least one output token.
    output: output !== null && output > 0 ? output : text !== '' || toolCalls.length > 0 ? roughTokens(text) : 1,
  };
  return { text, toolCalls, finishReason: finish, usage };
}

/** The OpenAI completion object: the answer, with the dialled id put back on it. */
export function openAICompletion(answer: Answer, dialed: string, id: string): Record<string, unknown> {
  const message: Record<string, unknown> = { role: 'assistant', content: answer.text === '' && answer.toolCalls.length > 0 ? null : answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({ id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }));
  }
  return {
    id,
    object: 'chat.completion',
    created: CREATED_AT,
    model: dialed,
    choices: [{ index: 0, message, finish_reason: answer.finishReason }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

function sse(event: string | null, data: unknown): string {
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  return event === null ? `data: ${payload}\n\n` : `event: ${event}\ndata: ${payload}\n\n`;
}

/**
 * The OpenAI event stream: a role chunk, the text (and any tool calls), exactly one finish chunk,
 * a usage chunk, then `[DONE]` last. The finish chunk carries no content after it.
 */
export function openAIStream(answer: Answer, dialed: string, id: string): string {
  const base = { id, object: 'chat.completion.chunk', created: CREATED_AT, model: dialed };
  let out = sse(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  if (answer.text !== '') out += sse(null, { ...base, choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }] });
  if (answer.toolCalls.length > 0) {
    const delta = answer.toolCalls.map((call, index) => ({
      index,
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
    out += sse(null, { ...base, choices: [{ index: 0, delta: { tool_calls: delta }, finish_reason: null }] });
  }
  out += sse(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: answer.finishReason }] });
  out += sse(null, {
    ...base,
    choices: [],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  });
  return `${out}data: [DONE]\n\n`;
}

function anthropicBlocks(answer: Answer): Record<string, unknown>[] {
  const blocks: Record<string, unknown>[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) blocks.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: isRecord(input) ? input : {} });
  }
  return blocks;
}

/** The Anthropic message object: the answer, with the dialled id put back on it. */
export function anthropicMessage(answer: Answer, dialed: string, id: string): Record<string, unknown> {
  return {
    id,
    type: 'message',
    role: 'assistant',
    model: dialed,
    content: anthropicBlocks(answer),
    stop_reason: stopReasonForAnthropic(answer.finishReason),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/**
 * The Anthropic event stream in the canonical order: message_start first, each content block
 * opened, deltaed and closed, then message_delta (stop reason and usage), then message_stop.
 */
export function anthropicStream(answer: Answer, dialed: string, id: string): string {
  const blocks = anthropicBlocks(answer);
  let out = sse(
    'message_start',
    {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model: dialed,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    },
  );
  blocks.forEach((block, index) => {
    if (block.type === 'tool_use') {
      out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} } });
      out += sse('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input ?? {}) },
      });
    } else {
      out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
      const text = typeof block.text === 'string' ? block.text : '';
      if (text !== '') out += sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text } });
    }
    out += sse('content_block_stop', { type: 'content_block_stop', index });
  });
  out += sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReasonForAnthropic(answer.finishReason), stop_sequence: null },
    usage: { output_tokens: answer.usage.output },
  });
  out += sse('message_stop', { type: 'message_stop' });
  return out;
}

/** GET /v1/models on the OpenAI wire. */
export function openAIModelList(): Record<string, unknown> {
  return {
    object: 'list',
    data: FREE_MODELS.map((model: FreeModel) => ({
      id: dialableId(model),
      object: 'model',
      created: CREATED_AT,
      owned_by: 'openrouter',
      free: true,
      context_length: model.context,
    })),
  };
}

/** GET /v1/models on the Anthropic wire: the same fourteen ids, once each. */
export function anthropicModelList(): Record<string, unknown> {
  const data = FREE_MODELS.map((model: FreeModel) => ({
    type: 'model',
    id: dialableId(model),
    display_name: model.display,
    created_at: '2025-01-01T00:00:00Z',
    free: true,
    context_length: model.context,
  }));
  return {
    data,
    has_more: false,
    first_id: data.length > 0 ? data[0].id : null,
    last_id: data.length > 0 ? data[data.length - 1].id : null,
  };
}

export type Wire = 'openai' | 'anthropic';

/** An error in the client's own wire shape. */
export function errorBody(wire: Wire, type: string, message: string): Record<string, unknown> {
  if (wire === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, param: null, code: null } };
}
