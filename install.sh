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
#   4. install + enable the systemd --user unit (if systemctl is available)
#   5. print the one ~/.omp/agent/config.yml edit that routes the judge role here
#
# Idempotent: safe to run repeatedly. Never pushes, never installs system packages,
# never touches a GPU, never fails if systemd is not present.

set -eu

REPO_URL=${JEV_REPO_URL:-https://github.com/<you>/jev-local.git}
INSTALL_DIR=${JEV_INSTALL_DIR:-$HOME/.local/share/jev-local}
EXT_DIR=${JEV_EXT_DIR:-$HOME/.omp/plugins/node_modules/jev-local}
UNIT_NAME=jev-style-decision.service
UNIT_DEST=${JEV_UNIT_DEST:-$HOME/.config/systemd/user/$UNIT_NAME}

say()  { printf '%s\n' "$*"; }
die()  { printf 'install: %s\n' "$*" >&2; exit 1; }

# ---- 1. get the source ---------------------------------------------------------
if [ -n "${JEV_LOCAL_DIR:-}" ]; then
    SRC=$(cd "$JEV_LOCAL_DIR" && pwd)
    say "== using local checkout: $SRC"
else
    if [ -d "$INSTALL_DIR/.git" ] || [ -f "$INSTALL_DIR/.git" ]; then
        say "== updating $INSTALL_DIR"
        git -C "$INSTALL_DIR" pull --ff-only || die 'update failed; resolve manually'
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

# ---- 4. install the systemd --user unit ---------------------------------------
# The committed unit uses %h and a documented default of ~/.local/share/jev-local.
# When installing from a checkout somewhere else (JEV_LOCAL_DIR/JEV_INSTALL_DIR),
# the installed copy pins ExecStart's WorkingDirectory and run.sh path to the
# checkout that was actually installed -- a unit pointing at a deleted directory
# is worse than no unit. The committed file itself never carries an absolute path.
say "== installing systemd user unit $UNIT_NAME"

if command -v systemctl >/dev/null 2>&1; then
    mkdir -p "$(dirname "$UNIT_DEST")"
    # Substitute the repo's documented default paths with the real install path.
    # sed, not a hardcoded path: the committed unit stays relocatable, and this
    # substitution is what pins the live unit to this checkout.
    UNIT_TMP="$UNIT_DEST.tmp.$$"
    sed -e "s|%h/.local/share/jev-local|$SRC|g" "$SRC/systemd/$UNIT_NAME" > "$UNIT_TMP"
    # Verify the resulting unit is at least syntactically loadable before it is put
    # in place: a half-written unit file in ~/.config/systemd/user breaks every
    # --user command on the machine, not just this service.
    if systemd-analyze verify "$UNIT_TMP" >/dev/null 2>&1; then
        mv "$UNIT_TMP" "$UNIT_DEST"
        say "   installed $UNIT_DEST"
    else
        # verify is advisory, not authoritative: it cannot resolve %h or the
        # runtime-only paths, so a failure here is reported but does not block.
        # (Verified on systemd 262: `systemd-analyze verify` on the committed unit
        # exits 0 with no output once the paths exist.)
        mv "$UNIT_TMP" "$UNIT_DEST"
        say "   installed $UNIT_DEST (systemd-analyze verify had warnings; see 'journalctl --user -xe')"
    fi
    systemctl --user daemon-reload
    say "   systemctl --user daemon-reload: ok"
    if systemctl --user enable --now "$UNIT_NAME" 2>/dev/null; then
        say "   systemctl --user enable --now $UNIT_NAME: ok"
    else
        # A failing enable --now is expected when the model is not downloaded yet:
        # the unit's preflight sha256's the GGUF and exits 2, which Restart=on-failure
        # turns into a restart loop until the start limit trips. Saying so beats a
        # silent success that leaves the operator with a loop.
        say "   WARNING: the unit did not start (service start failed)."
        say "   Most likely the model is not downloaded yet -- see 'Fetch the model' below."
        say "   It is enabled and will start on next login; or start it after fetching:"
        say "       systemctl --user start $UNIT_NAME"
        say "   Check the reason with:"
        say "       systemctl --user status $UNIT_NAME"
        say "       journalctl --user -u $UNIT_NAME -n 50 --no-pager"
    fi
else
    # No systemctl: say so, and give the exact manual steps. This is not an error:
    # the service runs perfectly well from service/run.sh.
    say "== systemctl not found -- skipping the systemd unit."
    say "   To install it manually, copy the unit and enable it:"
    say "       mkdir -p $HOME/.config/systemd/user"
    say "       cp $SRC/systemd/$UNIT_NAME $UNIT_DEST"
    say "       systemctl --user daemon-reload"
    say "       systemctl --user enable --now $UNIT_NAME"
    say "   (On a machine without systemd, run $SRC/service/run.sh instead.)"
fi

cat <<EOF

== installed.

Extension is now discoverable by omp (package.json declares omp.extensions).
To verify:
    ls -l $EXT_DIR

The decision service (loopback only, port 61021):
    systemctl --user status $UNIT_NAME      # if the unit was installed
    $SRC/service/run.sh                     # or just run it by hand

--------------------------------------------------------------- !!! READ THIS !!!
Routing the JUDGE role at the local model changes live agent behaviour: from
this moment every judge call goes to a local CPU model instead of the remote
router you were using before.

1. Back up your config first:
       cp ~/.omp/agent/config.yml ~/.omp/agent/config.yml.bak

2. Edit ~/.omp/agent/config.yml and set exactly this line (mind the two-space
   indent -- this lives under modelRoles:):

       modelRoles:
         judge: typesafe/jev-style-0.8b-decision-v3

3. Verify it took effect:
       omp config get modelRoles.judge
       # expect: typesafe/jev-style-0.8b-decision-v3

   (If your omp build has no 'config get', open the file and read the line.)

4. Revert at any time by restoring the backup:
       cp ~/.omp/agent/config.yml.bak ~/.omp/agent/config.yml

Why it is not done for you: the extension cannot set the judge role (there is no
setJudge on the extension API), and silently rewriting your judge role would
change agent behaviour the moment the extension loads.
--------------------------------------------------------------- !!! READ THIS !!!

Also note: the env var TYPESAFE_BASE_URL alone is NOT sufficient, because a
catalog row's explicit baseUrl beats the env var. The extension registers the
provider with an explicit baseUrl, which is what wins.

(The service reads JEV_GGUF, JEV_PORT, JEV_WORKERS; see $SRC/README.md.
Optional per-machine override file: ~/.config/jev-local/env)

Fetch the model (required before the service can answer anything):
    hf download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF --local-dir $SRC/assets \\
      "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf" tokenizer/tokenizer.json readout_config.json

Verify the service:
    curl -s http://127.0.0.1:61021/health
    curl -s http://127.0.0.1:61021/v1/models

Uninstall:
    systemctl --user disable --now $UNIT_NAME   # ignore the error if absent
    rm -f $UNIT_DEST
    systemctl --user daemon-reload
    rm $EXT_DIR
    rm -rf $INSTALL_DIR      # only if this installer cloned it
    # revert modelRoles.judge in ~/.omp/agent/config.yml if you changed it
EOF
