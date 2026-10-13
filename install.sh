#!/bin/sh
# jev-local installer (one command, idempotent).
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/jev-local/main/install.sh | sh
#   ./install.sh                 # install from a local clone
#   JEV_REPO_URL=https://github.com/<you>/jev-local.git ./install.sh
#   JEV_LOCAL_DIR=/path/to/clone ./install.sh     # install from a checkout on disk
#
# What it does:
#   1. clone or update the repo into ~/.local/share/jev-local (unless JEV_LOCAL_DIR)
#   2. build the CPU-only scorer if the binary is missing
#   3. link the extension into ~/.omp/plugins/node_modules/jev-local, which omp's
#      package.json manifest (omp.extensions / pi.extensions) auto-discovers
#   4. print how to run the service
#
# Idempotent: safe to run repeatedly. Never pushes, never installs system packages,
# never touches a GPU, never starts a long-lived service.

set -eu

REPO_URL=${JEV_REPO_URL:-https://github.com/<you>/jev-local.git}
INSTALL_DIR=${JEV_INSTALL_DIR:-$HOME/.local/share/jev-local}
EXT_DIR=${JEV_EXT_DIR:-$HOME/.omp/plugins/node_modules/jev-local}

say()  { printf '%s\n' "$*"; }
die()  { printf 'install: %s\n' "$*" >&2; exit 1; }

# ---- 1. get the source ---------------------------------------------------------
if [ -n "${JEV_LOCAL_DIR:-}" ]; then
    SRC=$(cd "$JEV_LOCAL_DIR" && pwd)
    say "== using local checkout: $SRC"
else
    if [ -d "$INSTALL_DIR/.git" ] || [ -f "$INSTALL_DIR/.git" ]; then
        say "== updating $INSTALL_DIR"
        git -C "$INSTALL_DIR pull --ff-only || die 'update failed; resolve manually'"
    else
        say "== cloning into $INSTALL_DIR"
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR" || die "clone failed from $REPO_URL"
    fi
    SRC=$INSTALL_DIR
fi

[ -f "$SRC/service/service.py" ]  || die "missing service/service.py in $SRC"
[ -f "$SRC/extension/index.ts" ] || die "missing extension/index.ts in $SRC"

# ---- 2. build the scorer (CPU only) if missing -------------------------------
if [ -x "$SRC/scorer/jev-gguf-scores" ]; then
    say "== scorer binary already present: $SRC/scorer/jev-gguf-scores"
elif command -v c++ >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; then
    say "== building CPU-only scorer (this clones llama.cpp and takes a few minutes)"
    sh "$SRC/scorer/build.sh" || die 'scorer build failed'
else
    say "== SKIPPED scorer build: c++ or cmake not found."
    say "   Install cmake + a C++ toolchain, then run: sh $SRC/scorer/build.sh"
fi

# ---- 3. link the extension so omp discovers it --------------------------------
say "== linking extension into $EXT_DIR"
mkdir -p "$(dirname "$EXT_DIR")"
if [ -L "$EXT_DIR" ] || [ -d "$EXT_DIR" ]; then
    rm -rf "$EXT_DIR"
fi
ln -s "$SRC/extension" "$EXT_DIR"
say "   $EXT_DIR -> $(readlink "$EXT_DIR" 2>/dev/null || echo "$SRC/extension")"

# ---- 4. what to do next -------------------------------------------------------
cat <<EOF

== installed.

Extension is now discoverable by omp (package.json declares omp.extensions).
To verify:
    ls -l $EXT_DIR

Run the decision service (loopback only, port 61021):
    $SRC/service/run.sh

Then, to route the judge role at the local model, edit
~/.omp/agent/config.yml and set:

    modelRoles:
      judge: typesafe/jev-style-0.8b-decision-v3

(The service reads JEV_GGUF, JEV_PORT, JEV_WORKERS; see $SRC/README.md.)

Uninstall:
    rm $EXT_DIR
    rm -rf $INSTALL_DIR      # only if this installer cloned it
EOF
