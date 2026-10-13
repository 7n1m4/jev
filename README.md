# jev-local

A local, **CPU-only** Jev-Style-0.8B-Decision-v3 service plus an oh-my-pi extension
that points the agent's `typesafe` judge provider at it. No GPU, no cloud calls for
judgments, no API key.

* The service (`service/`) exposes the oh-my-pi `systemone` / `alpha/decisions`
  contract over HTTP on loopback.
* The scorer (`scorer/`) reads decisions directly out of llama.cpp as
  `logit[' yes'] - logit[' no']` at each ` ->` slot — **no text is generated**.
* The extension (`extension/`) registers the provider with an explicit `baseUrl`,
  which is the mechanism that actually retargets the judge role.

> **Not pushed anywhere.** This repository has a single local commit and no remote.
> Point `install.sh`'s `JEV_REPO_URL` at your own fork/gist before using the
> one-line curl installer. (See "Uninstall" for how to remove it.)

## What it is

Jev-Style-0.8B-Decision-v3 is a *decision* model, not a chat model. It reads a
state plus a typed question (`noul` / `choice` / `score`) and answers. Its readout
spec (`readout_config.json`, `format: macjev-readout-v1`) defines

```
score_k  = logit[' yes'] - logit[' no']   at the k-th ' ->' slot
p        = softmax(scores / T)            T = fitted calibration temperature
```

which is a **scoring** computation. Plain chat-style generation does not give you a
decision, so this repo never generates text: it pre-fills the prompt and reads two
vocab rows of the logits at the slot positions. See `scorer/src/jev_scores.cpp`.

## Port 61021, and why

`61021` is **loopback-only** (`127.0.0.1`, never `0.0.0.0`).

The reason for this specific port: Linux's default ephemeral range is **32768–60999**,
so any port inside it can be claimed by the kernel for an outbound connection and
stolen out from under a server. `61021` sits just above that range, so a service
bound there keeps its port. (A port in the 50000s would work most of the time and
then break in a way that is very hard to diagnose.)

Override with `JEV_PORT`. The bind address is hard-coded and not overridable.

## Model provenance

| Property | Value |
|---|---|
| Model | `chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF` |
| File | `Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf` |
| Size | 529,296,864 bytes (~505 MiB) |
| sha256 | `0a19bc29bacc33e0d871146c8612b24dd14c2ed2e61cedeb7a928b0852628bac` |
| Base | Qwen3.5 0.8B (Apache-2.0), 24 blocks, 18 SSM + 6 full-attention layers |
| Params | 752.39 M |

Preflight refuses to bind unless the sha256 matches exactly, so a truncated or
substituted file can never be served.

The GGUF is **not** vendored here (it is 505 MiB and belongs in a HF cache, not a
git repo). Fetch it with:

```sh
hf download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF --local-dir ./assets \
  "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf" tokenizer/tokenizer.json readout_config.json
```

Then either keep it in `./assets/` (the default search path) or point `JEV_GGUF` at
it. `readout_config.json` must sit alongside the GGUF; it carries the calibration
temperature table the service needs.

## CPU-only guarantee

* The scorer hard-sets `mp.n_gpu_layers = 0`, so it cannot silently offload even if
  a flag or env var says otherwise.
* The build passes **no** `--device` and **no** `-DGGML_VULKAN`; every GPU backend
  is explicitly `=OFF`.
* Nothing in this repo references nvidia, Vulkan, CUDA, ROCm, OpenCL or torch.
* `scorer/build.sh` verifies the result: it fails the build if the linked binary
  contains a GPU library or GPU symbol.

For local development the repo expects a llama.cpp checkout (it clones one into
`./llama.cpp` if you do not have one) and a `py/` scorer module. If you already
have the `~/.local/share/jev` tree, the service finds it automatically.

## Run

```sh
./service/run.sh                 # http://127.0.0.1:61021
JEV_PORT=61022 ./service/run.sh
JEV_GGUF=/path/to.gguf ./service/run.sh
JEV_WORKERS=4 ./service/run.sh   # scorer subprocesses in the pool
```

Requires Python 3.11+ with `fastapi` and `uvicorn`. No pip install into system
python is needed if those are already present; otherwise use a venv.

### Verify

```sh
curl -s http://127.0.0.1:61021/health
curl -s http://127.0.0.1:61021/v1/models

curl -s -H 'Content-Type: application/json' http://127.0.0.1:61021/v1/systemone -d '{
  "model": "jev-style-0.8b-decision-v3",
  "state": "I was charged twice for my subscription this month.",
  "questions": {
    "q": { "type": "choice", "instructions": "Which team should handle this ticket?",
           "criteria": {"billing": "payments, invoices, refunds", "technical": "bugs and outages"} }
  }
}'
```

`POST /v1/systemone` and `POST /alpha/decisions` are the same handler. GET
endpoints: `/v1/models`, `/health`. Full reference lives in `service/README.md`.

## Install the oh-my-pi extension

```sh
./install.sh
```

This clones/updates the repo into `~/.local/share/jev-local`, builds the scorer if
missing, and symlinks the extension into
`~/.omp/plugins/node_modules/jev-local`. Because `extension/package.json` declares
an `omp.extensions` manifest, omp auto-discovers it; no config edit is required.

Verify the link:

```sh
ls -l ~/.omp/plugins/node_modules/jev-local
```

The extension registers the `typesafe` provider with `baseUrl` pointing at
`http://127.0.0.1:61021` (override with `JEV_BASE_URL`). Set the judge role by
editing `~/.omp/agent/config.yml`:

```yaml
modelRoles:
  judge: typesafe/jev-style-0.8b-decision-v3
```

### Why not just `TYPESAFE_BASE_URL=...`?

Because **a catalog row's explicit `baseUrl` beats the `TYPESAFE_BASE_URL`
environment variable.** Setting the env var alone is *not* sufficient to retarget
the judge role; the provider must be registered with an explicit baseUrl. That is
what the extension does.

### Judge role: read this before editing your config

Your existing `~/.omp/agent/config.yml` sets

```yaml
modelRoles:
  judge: kilo/typesafe/jev-router
```

which routes the judge role through OpenRouter to a *remote* typesafe-compatible
router. **The extension deliberately does NOT change this line** — there is no
`setJudge` on the extension API, and silently rewriting your judge role would
change agent behaviour the moment the extension loads. To use the local model,
edit `modelRoles.judge` yourself as shown above. Back up `~/.omp/agent/config.yml`
first.

## Confidence

`confidence = clip(1 − H / log k, 0, 1)` with `H = −Σ pᵢ log pᵢ`, matching the
reference runtime's own `concentration()`. One option → 1.0; uniform → 0.0; peaked →
near 1.0. Whole-distribution, so close runners-up genuinely lower it.

## Known limitations

* **Option chunking is not implemented.** A `choice` question whose options exceed
  the 2048-token head budget (roughly 21+ options) returns **HTTP 422** instead of
  being scored in chunks. This is the reference runtime's *non-default*
  `--no-split-options` behaviour: nothing is silently truncated, and the caller
  gets a deterministic error. The default reference path would chunk.
* **No state sharing across questions.** Each question re-decodes the state; the
  reference scorer shares one state via sequence copies.
* **`usage.output_tokens` counts verdict slots**, not generated tokens — there are
  no generated tokens.
* **Score-question keys are level index strings** (`"0"`, `"1"`, …).
* **Concurrency is pooled, not locked.** `JEV_WORKERS` scorer processes are used;
  each request takes a whole worker so two requests never interleave on one
  process's stdin (that would corrupt the JSON-lines protocol and return a
  confidently wrong answer).

## Uninstall

```sh
rm ~/.omp/plugins/node_modules/jev-local     # remove the extension link
rm -rf ~/.local/share/jev-local              # only if install.sh cloned it
# revert modelRoles.judge in ~/.omp/agent/config.yml if you changed it
```

Nothing else is installed: no systemd unit, no system package, no background
service that survives a reboot.

## Layout

```
service/    FastAPI service, preflight, pool, README.md, run.sh
scorer/     build.sh + src/jev_scores.cpp (CPU-only score readout)
extension/  index.ts + package.json (omp extension manifest)
install.sh  one-command, idempotent installer
```
