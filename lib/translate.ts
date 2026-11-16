// EXSALA HERMES MOD · powered by Gienetic
//
// lib/translate.ts — the Anthropic Messages body translated into a Chat Completions body, and
// the stop reasons translated between the two vocabularies.
//
// The free models have one route on the gateway (Chat Completions). An Anthropic-shaped client
// request therefore becomes a Chat Completions body here: system -> a system message, content
// blocks -> OpenAI parts, tools -> OpenAI tools, tool_use/tool_result <-> tool_calls/tool, with
// max_tokens and temperature carried across.

import { isRecord } from './models.ts';

export type Json = Record<string, unknown>;

/** The textual content of a string, or of an array of content blocks. */
export function textOfContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function jsonArguments(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

/** An Anthropic image block's source as an OpenAI image_url part. */
function imagePart(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = stringOr(source.media_type, 'image/png');
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** The system prompt (a string or text blocks) as one OpenAI system message content string. */
export function systemText(system: unknown): string {
  if (typeof system === 'string') return system;
  if (!Array.isArray(system)) return '';
  const parts: string[] = [];
  for (const block of system) {
    if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The tool names a client declared, in either wire's spelling. */
export function declaredToolNames(body: Json): string[] {
  const tools = body.tools;
  if (!Array.isArray(tools)) return [];
  const names: string[] = [];
  for (const tool of tools) {
    if (!isRecord(tool)) continue;
    const name = typeof tool.name === 'string' ? tool.name : isRecord(tool.function) && typeof tool.function.name === 'string' ? tool.function.name : '';
    if (name !== '') names.push(name);
  }
  return names;
}

/** Anthropic tools -> OpenAI tools. */
function openAITools(tools: unknown): Json[] | null {
  if (!Array.isArray(tools) || tools.length === 0) return null;
  const out: Json[] = [];
  for (const tool of tools) {
    if (!isRecord(tool) || typeof tool.name !== 'string') continue;
    const fn: Json = { name: tool.name };
    if (typeof tool.description === 'string') fn.description = tool.description;
    fn.parameters = isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} };
    out.push({ type: 'function', function: fn });
  }
  return out.length === 0 ? null : out;
}

/** Anthropic tool_choice -> OpenAI tool_choice. */
function openAIToolChoice(choice: unknown): unknown {
  if (typeof choice === 'string') return choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'any') return 'required';
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'none') return 'none';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

/**
 * Translate one Anthropic Messages request into the Chat Completions body the gateway's only
 * route takes. Anthropic-only fields are dropped; the fields both wires share are carried.
 */
export function anthropicToOpenAI(body: Json): Json {
  const messages: Json[] = [];
  const system = systemText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });

  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of incoming) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;

    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;

    if (role === 'assistant') {
      const texts: string[] = [];
      const calls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
        else if (block.type === 'tool_use') {
          const name = typeof block.name === 'string' ? block.name : '';
          if (name === '') continue;
          calls.push({
            id: stringOr(block.id, `call_${calls.length}`),
            type: 'function',
            function: { name, arguments: jsonArguments(block.input) },
          });
        }
      }
      const message: Json = { role: 'assistant', content: texts.length > 0 ? texts.join('') : calls.length > 0 ? null : '' };
      if (calls.length > 0) message.tool_calls = calls;
      messages.push(message);
      continue;
    }

    // A user turn: tool results become their own `tool` messages, other blocks one user message.
    const parts: Json[] = [];
    const results: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image') {
        const part = imagePart(block);
        if (part !== null) parts.push(part);
      } else if (block.type === 'tool_result') {
        results.push({
          role: 'tool',
          tool_call_id: stringOr(block.tool_use_id, 'call_0'),
          content: textOfContent(block.content),
        });
      }
    }
    for (const result of results) messages.push(result);
    if (parts.length > 0) {
      const anyImage = parts.some((p) => p.type === 'image_url');
      messages.push(anyImage ? { role: 'user', content: parts } : { role: 'user', content: parts.map((p) => stringOr(p.text, '')).join('') });
    }
  }

  const out: Json = { messages };
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : 1024;
  out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = openAITools(body.tools);
  if (tools !== null) out.tools = tools;
  const choice = openAIToolChoice(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  return out;
}

const OPENAI_TO_ANTHROPIC: Record<string, string> = {
  stop: 'end_turn',
  length: 'max_tokens',
  tool_calls: 'tool_use',
  function_call: 'tool_use',
  content_filter: 'refusal',
};

const ANTHROPIC_TO_OPENAI: Record<string, string> = {
  end_turn: 'stop',
  max_tokens: 'length',
  stop_sequence: 'stop',
  tool_use: 'tool_calls',
  refusal: 'content_filter',
  pause_turn: 'stop',
};

/** A Chat Completions finish reason in the Anthropic vocabulary (always a legal one). */
export function stopReasonForAnthropic(finish: string): string {
  return OPENAI_TO_ANTHROPIC[finish] ?? 'end_turn';
}

/** An Anthropic stop reason in the Chat Completions vocabulary (always a legal one). */
export function stopReasonForOpenAI(reason: string): string {
  return ANTHROPIC_TO_OPENAI[reason] ?? 'stop';
}
