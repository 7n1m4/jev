#!/bin/sh
# Build the CPU-only scorer from source against a shared-library libllama.
#
# Zero GPU backends: no --device, no -DGGML_VULKAN, every backend explicitly OFF.
# The scorer itself hard-locks mp.n_gpu_layers = 0 so it cannot silently offload.
#
#   sh build.sh                     # clone or update llama.cpp into ./llama.cpp
#   LLAMA_SRC=/path/to/llama.cpp sh build.sh    # use an existing checkout
#
# Output: <repo>/scorer/jev-gguf-scores
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
SC="$REPO/scorer"
OUT="$SC/jev-gguf-scores"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,12p' "$0"
    exit 0
fi

LLAMA_SRC=${LLAMA_SRC:-}
if [ -z "$LLAMA_SRC" ]; then
    if [ ! -d "$REPO/llama.cpp/.git" ]; then
        echo "== cloning llama.cpp (shallow)"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$REPO/llama.cpp"
    fi
    LLAMA_SRC="$REPO/llama.cpp"
fi
LC=$(cd "$LLAMA_SRC" && pwd)

# 1) shared-library libllama, CPU only
echo "== configuring shared libllama (CPU only, no GPU backends)"
cmake -S "$LC" -B "$LC/build-shared" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
      -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_MUSA=OFF -DGGML_METAL=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF

echo "== building shared libllama"
cmake --build "$LC/build-shared" --target llama -j "$(nproc 2>/dev/null || echo 4)"

# 2) the scorer, linked against that shared libllama
echo "== compiling scorer"
c++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
    -I"$LC/include" -I"$LC/ggml/include" -I"$LC/vendor" \
    "$SC/src/jev_scores.cpp" -o "$OUT" \
    -L"$LC/build-shared/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LC/build-shared/bin"

echo "== built $OUT"
echo "== verifying no GPU libraries are linked"
if ldd "$OUT" | grep -icE "\b(cuda|nvidia|vulkan|hip|rocm|opencl|amdgpu)\b" | grep -qv '^0$'; then
    echo "FAIL: GPU library linked"; exit 1
fi
n=$(nm -D "$OUT" 2>/dev/null | grep -icE "\b(cuda|vulkan|hip|nvml|rocm|opencl)\b" || true)
if [ "${n:-0}" != "0" ]; then
    echo "FAIL: $n GPU symbols in the scorer"; exit 1
fi
echo "== CPU-only check passed (0 GPU libs, 0 GPU symbols)"
