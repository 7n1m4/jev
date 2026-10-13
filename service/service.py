"""Jev-Style-0.8B-Decision-v3 as an oh-my-pi `systemone` / `alpha/decisions` provider.

CPU-only. The model never generates text: decisions are read as
score_k = logit[' yes'] - logit[' no'] at the k-th ' ->' slot, per
readout_config.json (macjev-readout-v1 / macjev-render-v1), and turned into
probabilities with softmax(scores / T) using the fitted calibration
temperatures.

Wire contract is oh-my-pi's; see README.md in this directory.

Handlers are plain `def` on purpose (not `async def`) so FastAPI runs them in its
threadpool: inference is blocking CPU work, and an `async def` handler would
block the event loop and serialise every concurrent client.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent            # .../jev-local/service
REPO_ROOT = HERE.parent                           # .../jev-local
SCORER_DIR = Path(os.environ.get("JEV_SCORER_DIR", REPO_ROOT / "scorer"))

# The python side of the scorer lives in the sibling jev/ tree during development
# and is vendored into this repo; prefer the vendored copy, then the env override.
_JEVPY_CANDIDATES = [
    REPO_ROOT / "scorer" / "py",
    REPO_ROOT.parent / "jev" / "scorer" / "py",
    Path(os.environ.get("JEV_SCORER_PY", "")),
]
for _p in _JEVPY_CANDIDATES:
    if _p and (_p / "jevreadout.py").is_file():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(HERE))                    # pool.py

# Model: JEV_GGUF wins; otherwise look next to the repo, then the default dev tree.
_MODEL_DEFAULT = REPO_ROOT / "assets" / "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf"
if not _MODEL_DEFAULT.exists():
    _MODEL_DEFAULT = REPO_ROOT.parent / "jev" / "assets" / "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf"
MODEL_PATH = Path(os.environ.get("JEV_GGUF", str(_MODEL_DEFAULT)))
SCORER_BIN = Path(os.environ.get("JEV_SCORER_BIN", str(SCORER_DIR / "jev-gguf-scores")))

_ASSETS_CANDIDATES = [MODEL_PATH.parent, REPO_ROOT.parent / "jev" / "assets"]
READOUT_CONFIG = next((p / "readout_config.json" for p in _ASSETS_CANDIDATES
                       if (p / "readout_config.json").is_file()), _ASSETS_CANDIDATES[0] / "readout_config.json")
MANIFEST = next((p / "manifest.json" for p in _ASSETS_CANDIDATES
                 if (p / "manifest.json").is_file()), _ASSETS_CANDIDATES[0] / "manifest.json")
MODEL_NAME = "jev-style-0.8b-decision-v3"

DEFAULT_PORT = 61021
HOST = "127.0.0.1"                     # loopback only; never 0.0.0.0
N_WORKERS = int(os.environ.get("JEV_WORKERS", "2"))
N_CTX = 32768                          # reference runtime context limit

EXPECTED_SHA256 = "0a19bc29bacc33e0d871146c8612b24dd14c2ed2e61cedeb7a928b0852628bac"

QTYPES = ("choice", "noul", "score")


class PreflightError(RuntimeError):
    """Preflight failure; the service exits non-zero rather than serving errors."""


# ---- logging ------------------------------------------------------------------
logger = logging.getLogger("jev-service")


def _plog(message: str) -> None:
    """Preflight output, flushed explicitly so it survives uvicorn's logging."""
    print(message, flush=True)


# ---- request validation -----------------------------------------------------------
def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{what} must be a non-empty string")
    return value


def _validate_request(body: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return (model, state, questions) or raise 422. Exactly the oh-my-pi shape."""
    if not isinstance(body, dict):
        raise HTTPException(422, "request body must be a JSON object")

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(422, "field 'model' must be a non-empty string")

    if "state" not in body:
        raise HTTPException(422, "field 'state' is required (string or JSON value)")
    state = body["state"]
    if isinstance(state, str):
        if not state.strip():
            raise HTTPException(422, "field 'state' string must be non-empty")
    elif not isinstance(state, (dict, list)):
        raise HTTPException(422, "field 'state' must be a string, object or array")

    questions = body.get("questions")
    if not isinstance(questions, dict):
        raise HTTPException(422, "field 'questions' must be an object keyed by question id")
    if not questions:
        raise HTTPException(422, "field 'questions' must contain at least one question")

    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise HTTPException(422, f"question {qid!r} must be an object")
        t = q.get("type")
        if t not in QTYPES:
            raise HTTPException(
                422, f"question {qid!r} has unknown type {t!r}; expected one of {list(QTYPES)}")
        ins = q.get("instructions")
        if isinstance(ins, str):
            if not ins.strip():
                raise HTTPException(422, f"question {qid!r}: 'instructions' must be a non-empty string")
        elif ins is None:
            raise HTTPException(422, f"question {qid!r}: 'instructions' is required")
        # object instructions are allowed by the client's validation; the renderer
        # serialises non-string values, so accept any non-None value.

        crit = q.get("criteria")
        if t == "choice":
            if crit is None or crit == {}:
                raise HTTPException(422, f"question {qid!r}: 'choice' needs a non-empty criteria map")
            if not isinstance(crit, dict):
                raise HTTPException(422, f"question {qid!r}: 'choice' criteria must be an object of label -> description")
            for label, desc in crit.items():
                if desc is not None and not isinstance(desc, str):
                    raise HTTPException(
                        422, f"question {qid!r} option {label!r}: description must be a string or null")
        elif t == "score":
            if not isinstance(crit, list) or len(crit) < 2:
                raise HTTPException(
                    422, f"question {qid!r}: 'score' criteria must be an array of >= 2 level descriptions")
            for i, lv in enumerate(crit):
                if lv is None or (isinstance(lv, str) and not lv.strip()):
                    raise HTTPException(422, f"question {qid!r} level {i}: descriptions must be non-empty")
        elif t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise HTTPException(422, f"question {qid!r}: 'noul' criteria must be null or an object")

    return model, state, questions


# ---- answers ---------------------------------------------------------------------
def concentration(probs: List[float]) -> float:
    """Confidence = 1 - normalised entropy of the answer distribution.

    Matches the reference runtime's `concentration()` (jev_style_decision_gguf.py:296):
        H = -sum p log p ;  confidence = clip(1 - H / log(k), 0, 1)
    so a single-option question is 1.0, a uniform k-way distribution is 0.0, and a
    peaked one is close to 1.0. Chosen over top-probability because it accounts for
    the whole distribution, so two close runners-up at the top actually reduce it.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return float(min(max(1.0 - ent / math.log(k), 0.0), 1.0))


def _answer_for(qtype: str, dec: Any) -> Dict[str, Any]:
    names = dec.names
    probs = dec.probabilities

    if qtype == "noul":
        # CONFIRMED from the reference runtime source: make_question()/option_names()
        # return the LITERAL ["false", "true"] for every noul question
        # (jev_style_decision_gguf.py:84-88), and render_options() renders them in
        # that same order (lines 141-145). So probabilities[1] is P(true).
        #
        # This is asserted, not assumed: a silent order flip would return a
        # confidently wrong "probability of yes" -- the same class of bug as the
        # n_outputs_max_per_seq=1 slot-dropping bug. Loud 500, never a wrong answer.
        if list(names) != ["false", "true"]:
            raise AssertionError(
                f"noul verdict slot order is {list(names)!r}, expected ['false', 'true']; "
                f"refusing to answer because P(true) would be read from the wrong slot"
            )
        if len(probs) != 2:
            raise AssertionError(
                f"noul returned {len(probs)} probabilities, expected 2")
        p_true = probs[1]
        if not math.isfinite(p_true) or not 0.0 <= p_true <= 1.0:
            raise AssertionError(f"noul P(true) is not a finite probability: {p_true!r}")
        return {"type": "noul", "noul": round(float(p_true), 6)}

    if qtype == "choice":
        prob_map = {n: round(float(p), 6) for n, p in zip(names, probs)}
        return {
            "type": "choice",
            "choice": dec.answer,
            "probabilities": prob_map,
            "confidence": round(concentration(probs), 6),
        }

    # score: expected level index = sum(i * p_i), fractional; keys are level strings.
    prob_map = {n: round(float(p), 6) for n, p in zip(names, probs)}
    expected = 0.0
    for i, p in enumerate(probs):
        expected += i * float(p)
    return {
        "type": "score",
        "score": round(float(expected), 6),
        "probabilities": prob_map,
        "confidence": round(round(concentration(probs), 6), 6),
    }


# ---- app ---------------------------------------------------------------------------
app = FastAPI(title="jev-style-decision", version="1.0.0")

_pool = None
_pool_lock = threading.Lock()
_state: Dict[str, Any] = {"requests": 0, "errors": 0, "started": time.time()}
_state_lock = threading.Lock()


class Unavailable(RuntimeError):
    """The scorer pool is unhealthy; surface as 500 so the client retries."""


def get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            from pool import ScorerPool

            _pool = ScorerPool(n=N_WORKERS, n_ctx=N_CTX, threads=16)
            _plog(f"preflight: scorer pool started with {N_WORKERS} workers")
    return _pool


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    get_pool()


@app.get("/health")
def health() -> Dict[str, Any]:
    """Unauthenticated liveness. Plain JSON, no auth, no model access."""
    return {"status": "ok", "model": MODEL_NAME, "uptime_s": round(time.time() - _state["started"], 3)}


@app.get("/v1/models")
def models(request: Request) -> JSONResponse:
    """Discovery. oh-my-pi only calls this when an apiKey is configured.

    The client accepts ANY bearer token and tolerates no auth, so no rejection here.
    The shape must be exactly {models: [{name, description, release_date}]} or
    oh-my-pi's discovery returns null and falls back.
    """
    return JSONResponse({
        "models": [
            {
                "name": MODEL_NAME,
                "description": (
                    "Jev-Style 0.8B Decision v3: on-device typed decision readout "
                    "(noul / choice / score) over llama.cpp Q4_K_M, CPU-only."
                ),
                "release_date": "2026-09-26",
            }
        ]
    })


def _decide_one(state: Any, qid: str, q: Dict[str, Any], category: Optional[str]) -> Tuple[Dict[str, Any], int, float]:
    """Score one question. Returns (answer_envelope, input_tokens, cxx_ms)."""
    from jevreadout import (
        Decision,
        InputBudgetError,
        QuestionError,
        option_names,
    )

    pool = get_pool()
    qtype = q["type"]
    instructions = q["instructions"]
    criteria = q.get("criteria")

    try:
        dec, _w = pool.decide(
            state,
            instructions,
            options=criteria,
            qtype=qtype,
            category=category,
        )
    except (InputBudgetError, QuestionError) as exc:
        # a user-input problem: 422, which the client does NOT retry
        raise HTTPException(422, f"question {qid!r}: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        # scorer died / internal failure: 500, which the client DOES retry
        with _state_lock:
            _state["errors"] += 1
        logger.error("scorer failure on question %r: %s", qid, exc, exc_info=True)
        raise HTTPException(500, f"inference failed for question {qid!r}: {exc}")

    # sanity: the answer count must match the option count the client sent back
    names = option_names({"t": qtype, "ins": str(instructions), "crit": criteria})
    if len(dec.probabilities) != len(names):
        raise HTTPException(500, f"question {qid!r}: scorer returned "
                                 f"{len(dec.probabilities)} probabilities for {len(names)} options")
    if not math.isfinite(dec.temperature) or dec.temperature <= 0.0:
        raise HTTPException(500, f"question {qid!r}: non-finite calibration temperature")

    # The assertions in _answer_for (notably the noul verdict-slot order) are
    # silent-wrong-answer guards, so they must surface as a STRUCTURED 500 the
    # client can parse -- an unhandled traceback would break the envelope and the
    # client would throw ProviderResponseError instead of retrying loudly.
    try:
        env = _answer_for(qtype, dec)
    except AssertionError as exc:
        with _state_lock:
            _state["errors"] += 1
        logger.error("INVARIANT VIOLATION on question %r: %s", qid, exc)
        raise HTTPException(500, f"internal invariant violated for question {qid!r}: {exc}")

    # The client compares its own question's type with the answer's type and throws
    # on any mismatch, so this is checked rather than assumed.
    if env.get("type") != qtype:
        raise HTTPException(500, f"question {qid!r}: answer type {env.get('type')!r} != question type {qtype!r}")

    return env, dec.input_tokens, dec.t_ms


def _handle(body: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    model, state, questions = _validate_request(body)

    answers: Dict[str, Any] = {}
    in_tokens = 0
    out_tokens = 0
    verdicts: List[str] = []

    for qid, q in questions.items():
        # each question is scored on its own request round-trip; category is a
        # per-request convention (a caller may send it at the top level, else None)
        category = body.get("category") if isinstance(body.get("category"), str) else None
        env, tin, cxx_ms = _decide_one(state, qid, q, category)
        answers[qid] = env
        in_tokens += tin
        # output_tokens in a decision readout is the number of scored verdict slots
        out_tokens += len(env.get("probabilities", {})) or 1
        verdicts.append(str(env.get("choice") if env.get("type") == "choice"
                            else env.get("answer", env.get("score", env.get("noul")))))

    elapsed = (time.perf_counter() - t0) * 1000.0
    with _state_lock:
        _state["requests"] += 1

    # input_tokens/output_tokens are optional for the client (non-finite -> 0) but
    # real numbers so operators can see the cost. Emit plain JSON-safe ints.
    usage = {"input_tokens": int(in_tokens), "output_tokens": int(out_tokens)}
    if not all(math.isfinite(v) and v >= 0 for v in usage.values()):
        usage = {"input_tokens": 0, "output_tokens": 0}

    logger.info(
        "POST decisions %d question(s) -> %s | %d in / %d out tok | %.1f ms",
        len(questions), ",".join(verdicts), in_tokens, out_tokens, elapsed,
    )

    return {"model": model, "answers": answers, "usage": usage}


# NOTE: plain `def`, not `async def` -- see the module docstring.
@app.post("/v1/systemone")
def systemone(body: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(_handle(body))


@app.post("/alpha/decisions")
def alpha_decisions(body: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(_handle(body))


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):  # noqa: ARG001
    """422 carries a structured {error:{type,message}} body the client parses."""
    if exc.status_code == 422:
        return JSONResponse(status_code=422, content={
            "error": {"type": "invalid_request_error", "message": exc.detail}
        })
    return JSONResponse(status_code=exc.status_code, content={
        "error": {"type": "server_error", "message": exc.detail}
    })


# ---- preflight ----------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def preflight() -> None:
    """Verify the model, the scorer binary and a live scorer probe, before binding.

    Order mirrors ~/embed/embedding_service.py: cheapest and most diagnostic first.
    A failure exits non-zero with a specific message rather than holding the port
    and serving errors.
    """
    # 1. model file: exists and hashes to the manifest value, so a truncated or
    #    substituted download can never be served.
    if not MODEL_PATH.is_file():
        raise PreflightError(f"model file missing: {MODEL_PATH}")
    size = MODEL_PATH.stat().st_size
    digest = _sha256(MODEL_PATH)
    if digest != EXPECTED_SHA256:
        raise PreflightError(
            f"model sha256 mismatch: {digest} (expected {EXPECTED_SHA256}); "
            f"{MODEL_PATH} is truncated or substituted"
        )
    _plog(f"preflight: model ok ({size} bytes, sha256 {digest[:16]}...)")

    # 2. readout config, because the temperature table and render format live there.
    if not READOUT_CONFIG.is_file():
        raise PreflightError(f"readout_config.json missing: {READOUT_CONFIG}")
    cfg = json.loads(READOUT_CONFIG.read_text())
    if cfg.get("format") != "macjev-readout-v1" or cfg.get("template") != "macjev-render-v1":
        raise PreflightError(
            f"unexpected readout format {cfg.get('format')!r}/{cfg.get('template')!r}")
    global_ = float(cfg["temperatures"]["global"])
    _plog(f"preflight: readout_config ok (global T={global_:.6f}, "
             f"{len(cfg['temperatures']['groups'])} fitted groups)")

    # 3. scorer binary, and it must answer a probe that proves the slot tokens.
    if not SCORER_BIN.is_file():
        raise PreflightError(f"scorer binary missing: {SCORER_BIN}")
    probe = subprocess.run(
        [str(SCORER_BIN), str(MODEL_PATH)],
        input='{"cmd":"quit"}\n',
        capture_output=True, text=True, timeout=900,
    )
    if probe.returncode != 0:
        raise PreflightError(
            f"scorer probe exited {probe.returncode}: {probe.stderr[-4000:]}")
    if "slot tokens verified" not in probe.stderr:
        raise PreflightError(
            "scorer probe did not verify slot tokens: "
            + (probe.stderr[-2000:] or "(no stderr)"))
    for ln in probe.stderr.splitlines():
        if "slot tokens verified" in ln:
            _plog(f"preflight: scorer probe ok -- {ln.strip()}")
    _plog("preflight: all checks passed")


# ---- entry point ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("JEV_PORT", DEFAULT_PORT))
    try:
        preflight()
    except PreflightError as exc:
        print(f"FATAL: startup preflight failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)

    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Loopback only: this box's 32768-60999 range is the ephemeral band, so binding
    # 0.0.0.0 would expose the model, and a 5xxxx port risks being stolen by an
    # outbound connect. 127.0.0.1:61021 is outside the ephemeral range.
    print(f"jev-service: binding http://{HOST}:{port}", flush=True)
    uvicorn.run(app, host=HOST, port=port, workers=1, log_level="info")
