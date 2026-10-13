# jev-service — Jev-Style-0.8B-Decision-v3 as an oh-my-pi decision provider

CPU-only HTTP wrapper around the Jev-Style-0.8B-Decision-v3 GGUF, exposing the
oh-my-pi `systemone` / `alpha/decisions` contract. **No text is generated.** Each
answer is read off the model as

```
score_k = logit[' yes'] - logit[' no']   at the k-th ' ->' slot
probabilities = softmax(scores / T)
```

per `readout_config.json` (`format: macjev-readout-v1`, `template: macjev-render-v1`,
`readout: verdict`). The two logits are the dot product of the final normed hidden
state at that slot with the tied embedding rows, in float32 — which is exactly what
`llama_get_logits_ith` returns, so the service reads two vocab rows and never decodes.

## Run

```sh
./run.sh                     # binds http://127.0.0.1:61021
```

`run.sh` execs `python3 service.py`. The repo is self-contained and relocatable: the
model is found in `../assets/` first, then a sibling `jev/assets/` tree, then
`JEV_GGUF`. The scorer binary is expected at `../scorer/jev-gguf-scores` (override
with `JEV_SCORER_BIN`, or the whole scorer dir with `JEV_SCORER_DIR`).

### Environment variables

| Variable    | Default                        | Meaning                                   |
|-------------|--------------------------------|-------------------------------------------|
| `JEV_PORT`  | `61021`                        | TCP port.                                 |
| `JEV_GGUF`  | `../assets/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf` | Model file. |
| `JEV_WORKERS`| `2`                           | Scorer subprocesses in the pool.          |
| `JEV_SCORER_DIR` | `../scorer`               | Directory holding the scorer binary.       |
| `JEV_SCORER_BIN` | `JEV_SCORER_DIR/jev-gguf-scores` | Scorer binary path.                 |

Bind address is **hard-coded `127.0.0.1`** (loopback only, never `0.0.0.0`): port
61021 was chosen because 32768–60999 is this box's ephemeral range, so a port in
the 50000s can be claimed by an outbound connect, while 61021 cannot.

## Endpoints

All POST bodies are `Content-Type: application/json`. No streaming, no session state.

### `POST /v1/systemone` and `POST /alpha/decisions`

Identical handlers; oh-my-pi's auth-gateway serves the same contract on both paths.

Request:
```json
{
  "model": "jev-style-0.8b-decision-v3",
  "state": "string or any JSON value",
  "questions": {
    "<id>": { "type": "noul|choice|score", "instructions": "string|object", "criteria": ... }
  }
}
```
`state` may be a plain string or arbitrary JSON (it is serialised like the reference
runtime: strings pass through, everything else `json.dumps`). `questions` is keyed by
caller-chosen ids; answers come back under the **same** ids. `criteria`:
* `choice`: object `{ label: description|null }` — named labels, description optional.
* `score`: **ordered** array of ≥ 2 level descriptions.
* `noul`: omitted, or `{ "true": ..., "false": ... }` to describe the outcomes.

Response `200`:
```json
{
  "model": "<echoed request model>",
  "answers": {
    "<id>": { "type": "noul",   "noul": 0.913 },
    "<id>": { "type": "choice", "choice": "billing", "probabilities": {"billing":0.977,"technical":0.004,"sales":0.019}, "confidence": 0.892 },
    "<id>": { "type": "score",  "score": 1.64, "probabilities": {"0":0.066,"1":0.345,"2":0.476,"3":0.114}, "confidence": 0.172 }
  },
  "usage": { "input_tokens": 365, "output_tokens": 10 }
}
```
Each answer's `type` is asserted equal to its question's `type` before responding.
`score` is the **probability-weighted level index**, so it may be fractional.
`usage.output_tokens` counts scored verdict slots (there is no text generation);
`cost` is never emitted (the model is local and free).

### `GET /v1/models`

Discovery, exact shape:
```json
{"models":[{"name":"jev-style-0.8b-decision-v3","description":"...","release_date":"2026-09-26"}]}
```
No auth is enforced — oh-my-pi accepts any bearer token and its discovery returns
`null` (and falls back) if the shape is wrong, so auth must not gate this.

### `GET /health`

`{"status":"ok","model":"...","uptime_s":123.4}` — plain JSON, no auth, no model access.

## Confidence formula

`confidence = clip(1 − H / log k, 0, 1)` where `H = −Σ pᵢ log pᵢ` is the Shannon
entropy of the answer distribution and `k` the number of options. This is the
reference runtime's own `concentration()` (`jev_style_decision_gguf.py:296`). A
single-option question yields 1.0, a uniform distribution 0.0, a peaked one near 1.0.
Chosen over raw top-probability because it accounts for the *whole* distribution: two
close runners-up at the top genuinely reduce it. `noul` has a scalar `noul` field
(P(true)) instead, matching the contract.

## Error semantics

| Condition                                                | Status | Retried by client? |
|----------------------------------------------------------|--------|--------------------|
| Malformed body / unknown question `type` / budget exceeded| `422`  | **No**             |
| Scorer died, internal failure, bad envelope              | `500`  | Yes, up to 3×      |

`422` and `500` both use `{"error":{"type":...,"message":...}}`. A user-input problem
is never reported as `500` — that would make the client retry a hopeless request.
`>21`-option (or over-budget) choices raise **422** rather than chunking, which is the
reference runtime's non-default `--no-split-options` behaviour: nothing is silently
truncated, and the caller gets a deterministic error it can act on. See
*Known limitations*.

## Preflight

`preflight()` runs **before** the port binds, mirroring `~/embed/embedding_service.py`:

1. Model file exists.
2. Model sha256 `== 0a19bc29bacc33e0d871146c8612b24dd14c2ed2e61cedeb7a928b0852628bac`
   (the manifest value) — catches a truncated or substituted download.
3. `readout_config.json` present with the expected `format`/`template`.
4. Scorer binary exists and answers a live probe; the probe must print
   `slot tokens verified: ' yes'=9542, ' no'=874`, proving the GGUF vocab agrees with
   `readout_config.json`'s slot-token ids.

Any failure exits `2` with a precise message. Nothing binds first.

## Concurrency

Requests are scored by a **pool of `JEV_WORKERS` scorer subprocesses** rather than a
single locked one. Each `DecisionRuntime` owns one subprocess and must not be shared
concurrently: its stdin is a single JSON-lines stream, and two interleaved writers
would corrupt the protocol and pair one request's answer with another's questions —
silently wrong answers, the worst failure for a decision service. Workers therefore
come from a free `queue.LifoQueue` (taking an element is atomic in CPython), each
also guarded by its own `threading.Lock` as a belt-and-braces check.

A worker that dies is restarted, and the request that hit it returns **500**, never a
wrong answer.

Measured on this box (7800X3D, single client thread, 50-token state): single-request
latency **139 ms**; 8 concurrent requests, wall 0.61 s, **8/8 correct, no
mis-pairing**, 13.2 req/s, per-request latency 161–605 ms. Peak RSS ~1.2 GiB per
scorer process.

## Known limitations

* **Option chunking is not implemented.** A choice question whose options exceed the
  2048-token head budget (or more than 21 options) raises `422` instead of being
  scored in chunks. This matches the reference runtime's `--no-split-options`
  behaviour; the default reference path would chunk. Nothing is truncated.
* **No state sharing across questions.** Each question re-decodes the state; the
  reference scorer shares one state across questions via sequence copies. A
  state-sharing cache is the obvious next optimisation.
* **`usage.output_tokens` counts verdict slots**, not generated tokens (there are
  none).
* **Score-question keys are level index strings** (`"0"`, `"1"`, …), per the contract.

## Files

* `service.py` — FastAPI app, preflight, validation, endpoints.
* `pool.py` — the scorer subprocess pool.
* `run.sh` — start script.
* `conc_test.py` — concurrency/mis-pairing probe.

## CPU-only guarantee

The scorer hard-sets `mp.n_gpu_layers = 0` regardless of any CLI flag. The shared
libllama it links was configured with `-DGGML_CUDA=OFF -DGGML_HIP=OFF
-DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_MUSA=OFF -DGGML_METAL=OFF`, and
`libllama.so`/`libggml*.so`/the scorer binary all have **zero** GPU symbols
(word-boundary `nm -D` match) and link no GPU libraries. No nvidia, Vulkan, CUDA or
torch code is involved anywhere. (A naive `grep -i hip` falsely matches the mangled
C++ `std::__cxx11::sub_match` — see `../scorer` notes.)
