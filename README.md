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
* `systemd/jev-style-decision.service` runs it as a `--user` unit, so the service
  survives logout and comes back after a reboot.

> **Not pushed anywhere yet.** This repository has a single local commit and no
> remote. Point `install.sh`'s `JEV_REPO_URL` at your own fork/gist before using
> the one-line curl installer. (See "Uninstall" for how to remove it.)

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
```

A real `POST /v1/systemone` covering all three question types at once —
copy-pasteable, and using the exact field names `_validate_request` accepts
(`model`, `state`, `questions` keyed by id; `type` + `instructions` required;
`criteria` is an object for `choice`, an ordered array of ≥2 for `score`,
omitted-or-object for `noul`):

```sh
curl -s -H 'Content-Type: application/json' http://127.0.0.1:61021/v1/systemone -d '{
  "model": "jev-style-0.8b-decision-v3",
  "state": "I was charged twice for my subscription this month and the refund has not arrived after two weeks.",
  "questions": {
    "q_noul":   { "type": "noul",   "instructions": "Does this need an escalation to a human agent?" },
    "q_choice": { "type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "payments, invoices, refunds", "technical": "bugs and outages", "sales": "plans and upgrades"} },
    "q_score":  { "type": "score",  "instructions": "How urgent is this ticket?",
                  "criteria": ["routine, can wait for the next business day", "needs attention within the day", "customer is blocked or losing money", "regulatory or legal exposure"] }
  }
}'
```

Actual response from the service on this box (whole request, 1.05 s wall):

```json
{
  "model": "jev-style-0.8b-decision-v3",
  "answers": {
    "q_noul":   { "type": "noul",   "noul": 0.712336 },
    "q_choice": { "type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.983966, "technical": 0.00587, "sales": 0.010164},
                  "confidence": 0.915613 },
    "q_score":  { "type": "score",  "score": 2.257545,
                  "probabilities": {"0": 0.038128, "1": 0.117337, "2": 0.393398, "3": 0.451138},
                  "confidence": 0.205012 }
  },
  "usage": { "input_tokens": 325, "output_tokens": 8 }
}
```

Note the shapes, because they differ per type: `noul` carries a scalar `noul`
(P(true)), `choice` carries the winning label plus the full distribution, and
`score` carries a **float** — the probability-weighted level index (2.26 here,
not an integer), with `probabilities` keyed by level *index string*. The `score`
confidence is low (0.21) because the top two levels are nearly tied (0.39 vs
0.45); that is the whole-distribution entropy doing its job.

`POST /v1/systemone` and `POST /alpha/decisions` are the same handler. GET
endpoints: `/v1/models`, `/health`. Full reference lives in `service/README.md`.

## Run as a systemd user service

`install.sh` installs `systemd/jev-style-decision.service` into
`~/.config/systemd/user/` and runs `daemon-reload` + `enable --now` for you. Do
it by hand with:

```sh
cp systemd/jev-style-decision.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jev-style-decision
```

The committed unit is path-agnostic: it pins `Environment=JEV_LOCAL_DIR` to
`~/.local/share/jev-local` (which is what `install.sh` clones into) and expands
`${JEV_LOCAL_DIR}` in `ExecStart`, so pointing the unit at a different checkout is
one line rather than editing the unit body:

```sh
systemctl --user edit jev-style-decision
# [Service]
# Environment=JEV_LOCAL_DIR=/some/other/checkout
```

Installing from a checkout somewhere else (`JEV_LOCAL_DIR=/some/path
./install.sh`) is handled by the installer, which substitutes the path into the
installed copy — the committed file itself never carries an absolute path.

It runs `JEV_WORKERS=1 JEV_PORT=61021`: each scorer worker is a separate
subprocess (~1.2 GiB RSS, see `service/README.md`), and the judge role sends one
decision at a time, so extra workers buy contention and memory rather than
throughput.

```sh
systemctl --user status jev-style-decision      # state, PID, memory
journalctl --user -u jev-style-decision -f      # the preflight is here
systemctl --user stop jev-style-decision
systemctl --user disable --now jev-style-decision
```

Two things worth knowing before editing it:

* **Preflight runs before the port binds.** A missing model, a sha256 mismatch, a
  missing `readout_config.json` or a failed scorer probe all exit non-zero with
  a precise reason, and `Restart=on-failure` retries with a graduated backoff.
  So a fresh install with no model downloaded is *expected* to sit in
  `activating (auto-restart)` until you fetch the GGUF — check
  `journalctl --user -u jev-style-decision -n 50` for which check failed.
* **MemoryMax is deliberately not set.** A ceiling is only as good as its
  measurement, and a ceiling measured on one box silently becomes too small
  elsewhere — where it turns a healthy daemon into a start-limit-hit loop. The
  unit's comment records this box's measured cgroup (`memory.current` ~881 MiB,
  memory.peak ~881 MiB, 0 OOM events) and the recipe for sizing one yourself.
  `OOMPolicy=stop` is set so that *if* you add a limit, the policy is
  kill-and-restart rather than throttle.

Optional per-machine path overrides go in an `EnvironmentFile` rather than in
the unit, so upgrading the repo never conflicts with your local edits:

```sh
mkdir -p ~/.config/jev-local
# ~/.config/jev-local/env, in KEY=VALUE form:
#   JEV_GGUF=/path/to/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf
#   JEV_SCORER_DIR=/home/you/.local/share/jev-local/scorer
```

`JEV_WORKERS` and `JEV_PORT` are pinned on the unit's `env` command line, and
`JEV_GGUF` / `JEV_SCORER_DIR` / `JEV_SCORER_BIN` left to that file.

## Install the oh-my-pi extension

```sh
./install.sh
```

This clones/updates the repo into `~/.local/share/jev-local`, builds the scorer if
missing, and symlinks the extension into
`~/.omp/plugins/node_modules/jev-local`. Because `extension/package.json` declares
an `omp.extensions` manifest, omp auto-discovers it; no config edit is required.
It also installs and enables the systemd user unit (see above).

Verify the link:

```sh
ls -l ~/.omp/plugins/node_modules/jev-local
```

The extension registers the `typesafe` provider with `baseUrl` pointing at
`http://127.0.0.1:61021` (override with `JEV_BASE_URL`).

### Judge role: the one manual edit

An extension cannot set the judge role — there is no `setJudge` on the extension
API. So it stays on whatever role your config already names until you edit it:

```sh
cp ~/.omp/agent/config.yml ~/.omp/agent/config.yml.bak   # 1. back up first
```

```yaml
# ~/.omp/agent/config.yml  (2-space indent; lives under modelRoles:)
modelRoles:
  judge: typesafe/jev-style-0.8b-decision-v3
```

Then verify:

```sh
omp config get modelRoles.judge      # expect: typesafe/jev-style-0.8b-decision-v3
```

**This changes live agent behaviour.** From that line on, every judge call goes
to a local CPU model on this machine instead of the remote router you were using.
Judging becomes free and offline, but also slower and much weaker than a large
remote model — which shows up as worse review and grading decisions, not as an
error you would notice. Revert with the one line:

```sh
cp ~/.omp/agent/config.yml.bak ~/.omp/agent/config.yml
```

or by putting the previous value back (before this edit it was
`judge: kilo/typesafe/jev-router`, routing through OpenRouter).

The extension deliberately does NOT perform this edit: silently rewriting your
judge role would change agent behaviour the moment the extension loads, on every
machine it is installed on, with no operator in the loop.

### Why not just `TYPESAFE_BASE_URL=...`?

Because **a catalog row's explicit `baseUrl` beats the `TYPESAFE_BASE_URL`
environment variable.** Setting the env var alone is *not* sufficient to retarget
the judge role; the provider must be registered with an explicit baseUrl. That is
what the extension does.

## Confidence

`confidence = clip(1 − H / log k, 0, 1)` with `H = −Σ pᵢ log pᵢ`, matching the
reference runtime's own `concentration()`. One option → 1.0; uniform → 0.0; peaked →
near 1.0. Whole-distribution, so close runners-up genuinely lower it. `noul`
answers carry a scalar `noul` (P(true)) instead of a confidence value.

## Known limitations

* **Option chunking is not implemented.** A `choice` question whose options exceed
  the 2048-token head budget (roughly 21+ options) returns **HTTP 422** instead of
  being scored in chunks. This is the reference runtime's *non-default*
  `--no-split-options` behaviour: nothing is silently truncated, and the caller
  gets a deterministic error. The default reference path would chunk.
* **Over-budget choice questions also return 422.** Same mechanism, same reason —
  the head budget is checked before scoring, so an over-long option set never
  reaches the model. There is no partial result and no chunked fallback.
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
systemctl --user disable --now jev-style-decision   # stop + drop the autostart
rm ~/.config/systemd/user/jev-style-decision.service
systemctl --user daemon-reload
rm ~/.omp/plugins/node_modules/jev-local            # the extension link
rm -rf ~/.local/share/jev-local                     # only if install.sh cloned it
# revert modelRoles.judge in ~/.omp/agent/config.yml if you changed it
```

The `disable --now` line is safe to run even when the service is not installed
(it just reports that it could not find the unit). No system package was
installed, and nothing outside `~/.config`, `~/.local/share` and the extension
link is touched — the only leftover to be aware of is `~/.config/jev-local/env`
if you created path overrides, which is inert once the service is gone.

## Layout

```
service/    FastAPI service, preflight, pool, README.md, run.sh
scorer/     build.sh + src/jev_scores.cpp (CPU-only score readout)
extension/  index.ts + package.json (omp extension manifest)
systemd/    jev-style-decision.service (--user unit)
install.sh  one-command, idempotent installer
```
