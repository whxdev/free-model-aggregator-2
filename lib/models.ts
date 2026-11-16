// EXSALA HERMES MOD · powered by Gienetic
//
// lib/models.ts — the sealed free catalogue, the catalogue document, and the deterministic
// quality/capability ranking used to pick the model that serves a request.
//
// Everything here is a pure function of (the request, the gateway answers already seen). The
// ratings, context windows and capability flags are the brief's own sealed table; they are the
// only inputs to the pick, so a replay of the same requests makes the same picks.

/** The prefix the gateway routes by; a catalogue id is `openrouter/` + the OpenRouter slug. */
export const DIAL_PREFIX = 'openrouter/';

export interface FreeModel {
  /** The OpenRouter slug, e.g. `nvidia/nemotron-3.5-super-vl:free`. */
  readonly slug: string;
  /** The provider's published name, kept for the Anthropic model list. */
  readonly display: string;
  /** The house rating, 0..2000, higher is stronger. */
  readonly rating: number;
  /** The model's window, in tokens. */
  readonly context: number;
  /** Whether the model takes a JSON response format. */
  readonly json: boolean;
  /** Whether the model takes tools. */
  readonly tools: boolean;
  /** Whether the model takes images. */
  readonly vision: boolean;
}

/**
 * The fourteen free models, in the brief's order. The brief's order is the tie-break: two models
 * with the same rating are ranked by it, so the pick never depends on Map iteration or on chance.
 */
export const FREE_MODELS: readonly FreeModel[] = Object.freeze([
  { slug: 'baidu/ernie-5.5-flash-preview:free', display: 'Baidu: ERNIE-5.5-Flash-Preview (free)', rating: 1150, context: 1024000, json: true, tools: true, vision: false },
  { slug: 'cohere/north-mini-code:free', display: 'Cohere: North Mini Code (free)', rating: 1200, context: 256000, json: false, tools: true, vision: false },
  { slug: 'dots-studio/dots-3-note-preview:free', display: 'Dots Studio: Dots3-Note Preview (free)', rating: 1150, context: 512000, json: true, tools: true, vision: true },
  { slug: 'google/gemma-4-26b-a4b-it:free', display: 'Google: Gemma 4 26B A4B (free)', rating: 1150, context: 262144, json: true, tools: true, vision: true },
  { slug: 'google/gemma-4-31b-it:free', display: 'Google: Gemma 4 31B (free)', rating: 1210, context: 262144, json: true, tools: true, vision: true },
  { slug: 'inclusionai/ling-3.2-flash:free', display: 'inclusionAI: Ling 3.2 Flash (free)', rating: 1150, context: 1048576, json: true, tools: true, vision: true },
  { slug: 'liquid/lfm-3-1.7b:free', display: 'LiquidAI: LFM3-1.7B (free)', rating: 1150, context: 262144, json: true, tools: true, vision: false },
  { slug: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', display: 'NVIDIA: Nemotron 3 Nano Omni (free)', rating: 1150, context: 256000, json: false, tools: true, vision: true },
  { slug: 'nvidia/nemotron-3-ultra-550b-a55b:free', display: 'NVIDIA: Nemotron 3 Ultra (free)', rating: 1285, context: 1000000, json: false, tools: true, vision: false },
  { slug: 'nvidia/nemotron-3.5-lightning:free', display: 'NVIDIA: Nemotron 3.5 Lightning (free)', rating: 1221, context: 1000000, json: false, tools: true, vision: false },
  { slug: 'nvidia/nemotron-3.5-super-vl:free', display: 'NVIDIA: Nemotron 3.5 Super VL (free)', rating: 1303, context: 1000000, json: true, tools: true, vision: true },
  { slug: 'particle-ai/particle-1:free', display: 'Particle AI: Particle-1 (free)', rating: 1150, context: 1048576, json: true, tools: true, vision: true },
  { slug: 'recursive/flint-v1.0:free', display: 'Recursive: Flint (free)', rating: 1150, context: 262144, json: true, tools: true, vision: false },
  { slug: 'thinkingmachines/inkling-2-small:free', display: 'Thinking Machines: Inkling 2 Small (free)', rating: 1475, context: 1048576, json: false, tools: true, vision: true },
]);

const BY_SLUG = new Map<string, FreeModel>(FREE_MODELS.map((m) => [m.slug, m]));

/** The id a request names and the gateway routes: `openrouter/<slug>`. */
export function dialableId(model: FreeModel): string {
  return `${DIAL_PREFIX}${model.slug}`;
}

/** A reported id without the gateway prefix; the provider echoes its answer that way. */
export function bareId(id: string): string {
  return id.startsWith(DIAL_PREFIX) ? id.slice(DIAL_PREFIX.length) : id;
}

/** The free model an id names, with or without the gateway prefix, or null. */
export function freeModelOf(id: unknown): FreeModel | null {
  if (typeof id !== 'string' || id === '') return null;
  return BY_SLUG.get(bareId(id)) ?? null;
}

/**
 * The catalogue document. `CATALOG_JSON` is the single serialisation: `--list` prints it and the
 * register body embeds it, so the two are byte for byte the same document.
 */
export const CATALOG_JSON: string = JSON.stringify({
  providers: [
    {
      id: 'openrouter',
      base_url: 'https://openrouter.ai/api/v1',
      key_env: 'OPENROUTER_API_KEY',
      wire: 'openai',
      models: FREE_MODELS.map((m) => ({ id: dialableId(m), free: true, context_length: m.context })),
    },
  ],
});

/** What a request needs from the model that serves it. */
export interface Required {
  readonly json: boolean;
  readonly tools: boolean;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * The capabilities a request demands, read off the (already translated) Chat Completions body:
 * a JSON mode request carries a response_format, a tools request declares at least one tool.
 */
export function requiredOf(body: Record<string, unknown>): Required {
  const format = body.response_format;
  const json = isRecord(format) && (format.type === 'json_object' || format.type === 'json_schema');
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  return { json, tools };
}

/**
 * The gateway answers seen so far, per model. This is the only mutable input to a pick: a model
 * that refused is demoted below the models that have not, so a rate-limited model cannot take
 * every later prompt down with it. A success clears the count again.
 */
export interface Health {
  readonly refusals: Map<string, number>;
}

export function newHealth(): Health {
  return { refusals: new Map<string, number>() };
}

export function noteRefusal(health: Health, model: FreeModel): void {
  health.refusals.set(model.slug, (health.refusals.get(model.slug) ?? 0) + 1);
}

export function noteSuccess(health: Health, model: FreeModel): void {
  health.refusals.delete(model.slug);
}

/**
 * A model steps aside only after it has refused three times without answering anything since: it
 * keeps its chance to recover from a transient rate limit, while a model that is plainly done for
 * the trial stops costing calls.
 */
const DEMOTE_AFTER = 3;

export function demoted(health: Health, model: FreeModel): boolean {
  return (health.refusals.get(model.slug) ?? 0) >= DEMOTE_AFTER;
}

/**
 * The candidates for one request, best first: every model that meets the request's capabilities,
 * the ones that have not refused twice ahead of the ones that have, then by rating, then by the
 * brief's order. `preferred` (a model the client named itself) is moved to the front when it can
 * serve the request.
 */
export function rankCandidates(required: Required, health: Health, preferred: FreeModel | null): FreeModel[] {
  const eligible: { model: FreeModel; index: number }[] = [];
  FREE_MODELS.forEach((model, index) => {
    if (required.json && !model.json) return;
    if (required.tools && !model.tools) return;
    eligible.push({ model, index });
  });
  eligible.sort((a, b) => {
    const da = demoted(health, a.model) ? 1 : 0;
    const db = demoted(health, b.model) ? 1 : 0;
    if (da !== db) return da - db;
    if (b.model.rating !== a.model.rating) return b.model.rating - a.model.rating;
    return a.index - b.index;
  });
  const ranked = eligible.map((e) => e.model);
  if (preferred !== null) {
    const at = ranked.findIndex((m) => m.slug === preferred.slug);
    if (at > 0) {
      ranked.splice(at, 1);
      ranked.unshift(preferred);
    }
  }
  return ranked;
}
