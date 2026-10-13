// Jev-Style-0.8B-Decision-v3 CPU-only score readout against libllama (shared build).
//
// Implements readout_config.json -> "score":
//   score_k = logit[' yes'] - logit[' no'] at the k-th ' ->' slot
// computed from the final normed hidden state at that slot and the tied embedding
// rows, float32. h_slot . (w_yes - w_no) == logits_k[9542] - logits_k[874], because
// llama.cpp computes logits as h . output.weight^T and this model ties output.weight
// to token_embd.weight. So we read only the two rows we need from llama_get_logits_ith.
//
// NO TEXT IS GENERATED. Only logits at slot positions are requested (batch.logits[rel]=1).
//
// Protocol: JSON on stdin (one request object per line) -> JSON on stdout (one result
// object per line). {"cmd":"quit"} exits. Model is argc[1].
//
// Request (rendered by the Python side, which owns tokenization):
//   {"ids":[<token ids>], "slots":[<absolute positions of ' ->' arrows, in order>],
//    "n_ctx":32768, "n_batch":2048, "threads":16}
// Slots must lie inside ids and in strictly increasing order.
//
// Response:
//   {"scores":[...per slot...], "verdict":<index of max score>, "t_ms":...,
//    "logits_rows":2, "n_vocab":..., "finite":true}

#include "llama.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

static const int YES_ID = 9542;   // " yes"
static const int NO_ID  =  874;   // " no"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: jev-gguf-scores <model.gguf>\n"); return 2; }
    const std::string model_path = argv[1];

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;             // hard-locked CPU, regardless of CLI flags
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "FATAL: failed to load model\n"); return 1; }

    llama_vocab const * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    {
        // encode " yes" / " no" through the model's own vocab and compare
        // (the reference renderer enforces exactly this check)
        int n_ok = 0;
        struct { const char * txt; int want; } chk[] = {{" yes", YES_ID}, {" no", NO_ID}};
        for (auto & c : chk) {
            llama_token scratch[64] = {0};
            const int n = llama_tokenize(vocab, c.txt, (int32_t) std::strlen(c.txt),
                                         scratch, 64, false, false);
            if (n == 1 && scratch[0] == c.want) n_ok++;
            else std::fprintf(stderr, "WARN: slot token %s -> %d tokens (want id %d)\n", c.txt, n, c.want);
        }
        if (n_ok != 2) std::fprintf(stderr, "WARN: slot token ids differ from readout_config.json\n");
        else std::fprintf(stderr, "slot tokens verified: ' yes'=%d, ' no'=%d\n", YES_ID, NO_ID);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = 32768;         // covers the 25,600-token input limit + head
    cp.n_batch      = 2048;          // prompt-processing chunk; slots always land in last chunk
    cp.n_ubatch     = 512;
    cp.n_seq_max    = 1;
    // We request logits at every ' ->' slot in one decode, so the per-decode output cap
    // must cover all of them (0 = n_batch default leaves per_seq at 1, which drops rows).
    cp.n_outputs_max        = 512;   // slots per decode (a 21+ option question is <512)
    cp.n_outputs_max_per_seq = 512;
    cp.embeddings   = false;         // logits path, not the embedding path
    cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "FATAL: failed to create context\n"); return 1; }

    std::string line;
    while (true) {
        std::fprintf(stderr, "READY\n");
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        json out;
        try {
            json req = json::parse(line);
            std::string cmd = req.value("cmd", std::string("score"));
            if (cmd == "quit") break;

            std::vector<llama_token> ids = req.at("ids").get<std::vector<llama_token>>();
            std::vector<int> slots = req.at("slots").get<std::vector<int>>();
            const int n = (int) ids.size();
            if (n == 0) throw std::runtime_error("empty ids");
            if (slots.empty()) throw std::runtime_error("no slots");
            for (size_t i = 0; i < slots.size(); ++i) {
                if (slots[i] < 0 || slots[i] >= n - 1)   // slot is an arrow; must be non-final
                    throw std::runtime_error("slot position outside the question part");
                if (i && slots[i] <= slots[i-1])
                    throw std::runtime_error("slots must be strictly increasing");
            }
            if ((uint32_t) n > cp.n_ctx)
                throw std::runtime_error("input exceeds n_ctx=" + std::to_string(cp.n_ctx) + "; nothing truncated");

            llama_memory_clear(llama_get_memory(ctx), true);
            out["scores"] = json::array();
            out["n_vocab"] = n_vocab;

            // Chunked prefill; only the final chunk may contain a slot (we only request
            // logits there), so reject a slot that lands in a non-final chunk.
            double t0 = (double) llama_time_us() * 1e-3;
            bool finite = true;
            for (int start = 0; start < n; ) {
                const int len = std::min((int) cp.n_batch, n - start);
                const bool last = (start + len == n);
                llama_batch batch = llama_batch_init(len, 0, 1);
                batch.n_tokens = len;
                for (int j = 0; j < len; ++j) {
                    batch.token[j]   = ids[start + j];
                    batch.pos[j]     = start + j;
                    batch.n_seq_id[j]= 1;
                    batch.seq_id[j][0]=0;
                    batch.logits[j]  = 0;
                }
                std::vector<int> hit;
                for (int s : slots) {
                    const int rel = s - start;
                    if (rel >= 0 && rel < len) {
                        if (!last)
                            throw std::runtime_error("slot inside a non-final chunk; question longer than n_batch");
                        batch.logits[rel] = 1;
                        hit.push_back(rel);
                    }
                }
                if (hit.empty()) throw std::runtime_error("no slot in final chunk");
                if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
                llama_batch_free(batch);

                // For each requested slot, llama_get_logits_ith takes the BATCH TOKEN INDEX,
                // not a row number (llama-context.cpp: output_resolve_row rejects a token
                // whose batch.logits[i] was not set). So index by the position in `hit`.
                for (size_t k = 0; k < hit.size(); ++k) {
                    const float * lg = llama_get_logits_ith(ctx, (int32_t) hit[k]);
                    if (!lg) throw std::runtime_error("no logits at requested slot");
                    const double s = (double) lg[YES_ID] - (double) lg[NO_ID];
                    if (!std::isfinite(s)) finite = false;
                    double clean = std::isfinite(s) ? s : 0.0;
                    out["scores"].push_back(clean);
                    out["raw"].push_back({{"yes", (double) lg[YES_ID]}, {"no", (double) lg[NO_ID]}});
                }
                start += len;
            }
            double t1 = (double) llama_time_us() * 1e-3;
            out["finite"]  = finite;
            out["t_ms"]    = t1 - t0;
            out["n_tokens"]= n;
            out["n_slots"] = (int) slots.size();
            // verdict = argmax score (ties -> lowest index, matching the reference softmax order)
            int best = 0;
            for (size_t i = 1; i < out["scores"].size(); ++i)
                if (out["scores"][i].get<double>() > out["scores"][best].get<double>()) best = (int) i;
            out["verdict"] = best;
        } catch (const std::exception & e) {
            out = json::object();
            out["error"] = e.what();
        }
        std::printf("%s\n", out.dump().c_str());
        std::fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
