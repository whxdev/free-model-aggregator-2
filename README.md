[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/whxdev/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/whxdev/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# Free Model Aggregator 02

## What it is

One Node 22 + TypeScript server (standard library only, no npm packages) that answers both the
OpenAI Chat Completions wire and the Anthropic Messages wire, fronting the 14 free OpenRouter
models and dialling them through the trial gateway.

Routing is per prompt and capability-aware: a `response_format` demands a JSON-capable model and a
non-empty `tools` array demands a tools-capable one, then the highest house rating wins. So JSON
prompts go to the strongest JSON model (`nvidia/nemotron-3.5-super-vl:free`, 1303) and everything
else to the strongest overall (`thinkingmachines/inkling-2-small:free`, 1475).

A refusal (4xx/5xx) records `{status, code, model}` and fails over to the next-best capable model,
up to four attempts with a fixed `250 ms * attempt` delay; the trail is returned on
`x-makefun-upstream-failures`. Routing is deterministic — no `Date.now()`, no `Math.random()` — so
the same requests against the same gateway answers replay to the same picks.

## How to run

```sh
node main.ts           # listen, register, serve
node main.ts --list    # print the catalogue document and exit
```

## License

MIT — see [LICENSE](LICENSE).
