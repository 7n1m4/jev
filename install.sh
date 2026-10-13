#!/bin/sh
# jev-local installer (one command, idempotent).
#
#   curl -fsSL https://raw.githubusercontent.com/7n1m4/jev/main/install.sh | sh
#   ./install.sh                 # install from a local clone
#   JEV_REPO_URL=https://github.com/7n1m4/jev.git ./install.sh
#   JEV_LOCAL_DIR=/path/to/clone ./install.sh     # install from a checkout on disk
#
# What it does:
#   1. clone or update the repo into ~/.local/share/jev-local (unless JEV_LOCAL_DIR)
#   2. preflight: refuse to continue if the model or the scorer's python module
#      is missing (both gitignored, both required) -- BEFORE the expensive build
#   3. build the CPU-only scorer if the binary is missing
#   4. link the extension with `omp plugin link`, which registers it in
#      omp-plugins.lock.json -- a bare symlink is silently skipped by omp
#   5. install + enable the systemd --user unit (if systemctl is available),
#      writing a MemoryMax measured on THIS machine's cgroup
#   6. print the one ~/.omp/agent/config.yml edit that routes the judge role here
#
# Idempotent: safe to run repeatedly. Never pushes, never installs system packages,
# never touches a GPU, never fails if systemd is not present.

set -eu

REPO_URL=${JEV_REPO_URL:-https://github.com/7n1m4/jev.git}
INSTALL_DIR=${JEV_INSTALL_DIR:-$HOME/.local/share/jev-local}
EXT_DIR=${JEV_EXT_DIR:-$HOME/.omp/plugins/node_modules/jev-local}
UNIT_NAME=jev-style-decision.service
SVC_NAME=$UNIT_NAME          # unit systemctl is asked about (differs when a .new name is installed)
UNIT_DEST=${JEV_UNIT_DEST:-$HOME/.config/systemd/user/$UNIT_NAME}

say()  { printf '%s\n' "$*"; }
die()  { printf 'install: %s\n' "$*" >&2; exit 1; }

# True when a path is inside a directory that systemd --user actually loads units
# from. Used to scope the "do not clobber a running unit" guard so that a scratch
# JEV_UNIT_DEST is treated as an ordinary file rather than as the live unit.
in_systemd_unit_dir() {
    case "${1:-}" in
        /etc/systemd/user/*|/usr/lib/systemd/user/*|/lib/systemd/user/*|/run/systemd/user/*)
            return 0 ;;
    esac
    # $HOME/.config/systemd/user, resolved both ways because XDG_CONFIG_HOME may
    # be unset and ~ may be a symlink.
    _base="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    case "$1" in
        "$_base"/*) return 0 ;;
    esac
    if [ -n "${XDG_CONFIG_HOME:-}" ]; then
        case "$1" in
            "$HOME"/.config/systemd/user/*) return 0 ;;
        esac
    fi
    return 1
}

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

# ---- 2. preflight: the pieces a fresh clone does NOT have --------------------
# assets/, scorer/py/ and scorer/jev-gguf-scores are all gitignored on purpose
# (the model is 505 MiB, the scorer is a build output, py/ is upstream source),
# so a fresh clone can pass "git clone" and still be unrunnable. The hard
# requirement is the MODEL, because preflight() sha256's it before binding: a
# unit installed without it just sits in a restart loop until the start limit
# trips, which is a confusing way to learn about a missing file. So stop HERE,
# before anything expensive -- and specifically before the multi-minute scorer
# build -- and say exactly what to fetch.
#
# scorer/py/ provides jevreadout.py, which service.py imports at module level --
# without it the app dies with ModuleNotFoundError: No module named
# 'jevreadout'. That is also a hard stop, not a warning.
say "== preflight: model + scorer python module"

if ! ls "$SRC"/assets/*.gguf >/dev/null 2>&1; then
    cat >&2 <<EOF
install: MISSING MODEL. $SRC/assets has no .gguf file, and the service's
          preflight refuses to bind without one (it sha256-verifies it).

Fetch it (needs the huggingface CLI):

    hf download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF --local-dir $SRC/assets \\
      "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf" tokenizer/tokenizer.json readout_config.json

or point the service at an existing copy with JEV_GGUF -- put it in
~/.config/jev-local/env:

    JEV_GGUF=/path/to/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf

(Nothing was installed. Fix the model, then re-run this installer.)
EOF
    exit 1
fi
say "   model: $(ls "$SRC"/assets/*.gguf | head -1)"

# tokenizer.json is a SEPARATE silent failure from the two above, and it is worse
# for being discovered late: the preflight checks the model's sha256 and the
# readout config, both of which pass without it, so the unit starts and binds and
# only the first real request fails -- with a 500 whose cause is
# AttributeError: 'NoneType' object has no attribute 'encode', because
# jevreadout._load_tokenizer() swallowed the exception and returned None
# (scorer/py/jevreadout.py:421-428). Found by running a fresh clone end to end.
# So check it here, where the operator can still see it.
if [ ! -f "$SRC/assets/tokenizer/tokenizer.json" ]; then
    cat >&2 <<EOF
install: MISSING TOKENIZER. $SRC/assets/tokenizer/tokenizer.json not found.

Without it the unit starts and binds, and then EVERY request fails with HTTP 500
(AttributeError: 'NoneType' object has no attribute 'encode'), because the
scorer's tokenizer loader fails silently. Fetch it with the same command:

    hf download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF --local-dir $SRC/assets \\
      "Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf" tokenizer/tokenizer.json readout_config.json

(Nothing was installed. Fix the tokenizer, then re-run this installer.)
EOF
    exit 1
fi
say "   tokenizer: $SRC/assets/tokenizer/tokenizer.json"

if [ ! -f "$SRC/scorer/py/jevreadout.py" ]; then
    # The dev tree is a legitimate home for this module (it is where the service
    # looks second), so point at it as a convenience -- but do NOT symlink it
    # silently: a link into someone's working tree is a trap the next rebuild
    # will spring.
    _HINT=""
    for _cand in "$HOME/work/jev/scorer/py" "$HOME/jev/scorer/py"; do
        if [ -f "$_cand/jevreadout.py" ]; then
            _HINT="   (this machine has it at $_cand; the service finds it
    automatically if JEV_SCORER_PY points there, or just copy/symlink it.)"
            break
        fi
    done
    cat >&2 <<EOF
install: MISSING SCORER MODULE. $SRC/scorer/py/jevreadout.py not found, and
          service.py imports it at module level, so the service cannot start.
$_HINT

The reference source lives in the scorer's own repository; see the scorer
section of $SRC/README.md.

(Nothing was installed. Fix the module, then re-run this installer.)
EOF
    exit 1
fi
say "   scorer module: $SRC/scorer/py/jevreadout.py"

# ---- 3. build the scorer (CPU only) if missing -------------------------------
# Last of the cheap gates: now that the model and module are known present, spend
# the minutes on the build. Without c++/cmake it is a warning, not a failure --
# the service cannot score without it, but the extension and the unit are still
# worth having, and the operator gets the exact command to run.
if [ -x "$SRC/scorer/jev-gguf-scores" ]; then
    say "== scorer binary already present: $SRC/scorer/jev-gguf-scores"
elif command -v c++ >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; then
    say "== building CPU-only scorer (this clones llama.cpp and takes a few minutes)"
    sh "$SRC/scorer/build.sh" || die 'scorer build failed'
else
    say "== SKIPPED scorer build: c++ or cmake not found."
    say "   Install cmake + a C++ toolchain, then run: sh $SRC/scorer/build.sh"
fi

# ---- 4. link the extension so omp discovers it --------------------------------
# BUG (found live, silent): a bare `ln -s` into node_modules is NOT enough.
# getEnabledPlugins (packages/coding-agent/src/extensibility/plugins/loader.ts:244)
# takes the union of package.json#dependencies and omp-plugins.lock.json#plugins,
# and a lockfile-only entry is kept ONLY when the node_modules entry is a symlink
# (loader.ts:169). A bare ln -s writes no lockfile entry, so the plugin is
# filtered out as a "stale lockfile entry" and never loads -- `omp plugin list`
# does not show it and the provider is never registered, with no error anywhere.
#
# `omp plugin link <path>` is the supported path: it writes BOTH the symlink and
# the lockfile entry (manager.ts:902-956). It is idempotent -- it rm -rf's the
# link path and re-creates it -- verified by running it three times on this box
# and diffing the lockfile byte-for-byte. It also requires the extension's
# package.json to carry a "name", which extension/package.json does.
say "== linking extension for omp ($EXT_DIR)"

if command -v omp >/dev/null 2>&1; then
    say "   omp found on PATH: using 'omp plugin link' (writes the lockfile entry)"
    # From the extension's own directory, so the lockfile records an absolute
    # path rather than a relative one (manager.link resolves against cwd).
    if (cd "$SRC/extension" && omp plugin link .) >/dev/null 2>&1; then
        say "   $EXT_DIR -> $(readlink "$EXT_DIR" 2>/dev/null || echo '(see omp')"
        say "   lockfile: $HOME/.omp/plugins/omp-plugins.lock.json now records $(basename "$EXT_DIR")"
    else
        say "   WARNING: 'omp plugin link' failed; falling back to a bare symlink."
        say "   The extension MAY not be discovered -- verify with: omp plugin list"
        mkdir -p "$(dirname "$EXT_DIR")"
        rm -rf "$EXT_DIR"
        ln -s "$SRC/extension" "$EXT_DIR"
        say "   $EXT_DIR -> $SRC/extension"
    fi
else
    # No omp binary: the symlink is still correct and the extension works once
    # omp is installed and `omp plugin link` is run. Say so loudly, because this
    # is exactly the silent-no-load case above.
    say "   WARNING: 'omp' is not on PATH, so the lockfile entry cannot be written."
    say "   The symlink is in place, but omp will NOT load this extension until"
    say "   you run:  omp plugin link $SRC/extension"
    say "   Then verify with: omp plugin list"
    mkdir -p "$(dirname "$EXT_DIR")"
    if [ -L "$EXT_DIR" ] || [ -d "$EXT_DIR" ]; then
        rm -rf "$EXT_DIR"
    fi
    ln -s "$SRC/extension" "$EXT_DIR"
    say "   $EXT_DIR -> $SRC/extension"
fi

# ---- 5. install the systemd --user unit ---------------------------------------
# The committed unit uses %h and a documented default of ~/.local/share/jev-local.
# When installing from a checkout somewhere else (JEV_LOCAL_DIR/JEV_INSTALL_DIR),
# the installed copy pins ExecStart's WorkingDirectory and run.sh path to the
# checkout that was actually installed -- a unit pointing at a deleted directory
# is worse than no unit. The committed file itself never carries an absolute path.
#
# ---- MemoryMax: measured here, not carried in the repo -----------------------
# The committed unit has no MemoryMax line, on purpose: a ceiling measured on one
# box is exactly what a public repo must not hard-code. systemd cannot take it
# from an env file either -- MemoryMax=${VAR} is rejected with
# "Invalid memory limit, ignoring" and silently becomes infinity (verified on
# systemd 262 with a probe unit), because size-parsing directives are evaluated
# before the environment is applied. So the number is generated HERE, from this
# machine's own measurement.
#
# The cgroup counter is what MemoryMax is enforced against, and it runs ~500 MiB
# BELOW the ps VmHWM sum: the scorer's Private_Clean pages (the tail of the 505
# MiB GGUF) are evictable and the kernel drops them first. Sizing off ps RSS is
# therefore ~1.5x too loose, and sizing off memory.peak alone ignores the ~73 MiB
# kernel charge (stacks, page tables, slab) that ps cannot see. So: peak +
# 73 MiB, then a generous allowance for those file-backed pages coming back under
# external pressure, rounded up to a whole multiple of 16 MiB.
#
# !! HAZARD, and it is not hypothetical -- it happened during this installer's own
# end-to-end test. Step 5 overwrote
# ~/.config/systemd/user/jev-style-decision.service while the service was ACTIVE.
# The running process survived (systemd had already loaded the old fragment), but
# the file on disk then pointed ExecStart at the developer checkout, so the NEXT
# restart would have started the wrong tree. So an existing unit that is active is
# never silently replaced: it is backed up and the new one is written next to it
# as `<unit>.new` for the operator to swap in. JEV_UNIT_DEST redirects the whole
# destination, which is how a non-production install avoids all of this.
#
# UNIT_NAME is the name systemd is asked about; UNIT_SRC is the repo file it comes
# from. They are deliberately separate, because the renamed .new unit is not a
# repo filename.
UNIT_SRC=jev-style-decision.service

# The guard. Renames only the DESTINATION (and the name asked of systemctl), never
# the repo filename, so the substitution below still reads a real file.
#
# Scoped to a real systemd unit directory on purpose. Asking "is this name active"
# alone is not enough: with JEV_UNIT_DEST pointed at a scratch file -- which is the
# supported way to test this installer without touching the live box -- the name is
# still active (the live service holds it) and the guard would redirect a scratch
# install to <unit>.new for no reason. So the destination must be somewhere systemd
# actually loads units from AND the name must be active; a scratch file is simply a
# file, and a normal install may replace it freely.
if [ -f "$UNIT_DEST" ] && in_systemd_unit_dir "$UNIT_DEST" && \
   systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
    _bak="$UNIT_DEST.bak.$(date +%Y%m%d-%H%M%S)"
    cp -p "$UNIT_DEST" "$_bak"
    say "== $UNIT_NAME is ACTIVE right now -- not replacing it."
    say "   existing unit backed up to: $_bak"
    say "   new unit written to:      $UNIT_DEST.new"
    UNIT_DEST="$UNIT_DEST.new"
    SVC_NAME="${UNIT_DEST##*/}"          # what systemctl is asked about from here on
    say "   (rename it to $SVC_NAME and restart when ready, or delete it"
    say "    to keep the running one)"
fi
say "== installing systemd user unit $SVC_NAME"

if command -v systemctl >/dev/null 2>&1; then
    mkdir -p "$(dirname "$UNIT_DEST")"
    [ -f "$SRC/systemd/$UNIT_SRC" ] || die "missing systemd/$UNIT_SRC in $SRC"
    # Substitute the repo's documented default paths with the real install path.
    # sed, not a hardcoded path: the committed unit stays relocatable, and this
    # substitution is what pins the live unit to this checkout.
    #
    # The pattern is exact enough to be safe: it matches only the install root
    # "%h/.local/share/jev-local". The unit's override file path is
    # "%h/.config/jev-local/env" (.config, not .local/share), which the pattern
    # does not touch -- so a relocating install leaves the EnvironmentFile knob
    # pointing at the user's real env file. Verified by substitution before commit.
    UNIT_TMP="$UNIT_DEST.tmp.$$"
    sed -e "s|%h/.local/share/jev-local|$SRC|g" "$SRC/systemd/$UNIT_SRC" > "$UNIT_TMP"
    # Verify the resulting unit is at least syntactically loadable before it is put
    # in place: a half-written unit file in ~/.config/systemd/user breaks every
    # --user command on the machine, not just this service.
    if systemd-analyze verify "$UNIT_TMP" >/dev/null 2>&1; then
        mv "$UNIT_TMP" "$UNIT_DEST"
        say "   installed $UNIT_DEST"
    else
        # verify is advisory, not authoritative: it cannot resolve %h or the
        # runtime-only paths, so a failure here is reported but does not block.
        mv "$UNIT_TMP" "$UNIT_DEST"
        say "   installed $UNIT_DEST (systemd-analyze verify had warnings; see 'journalctl --user -xe')"
    fi
    systemctl --user daemon-reload
    say "   systemctl --user daemon-reload: ok"

    # Size MemoryMax from this machine's cgroup, then re-inject it into the file.
    # Only done AFTER a start: before the first run there is no peak to measure,
    # and a peak read off a dead unit would be meaningless.
    _peak_bytes=""
    if systemctl --user is-active --quiet "$SVC_NAME" 2>/dev/null; then
        _peak_bytes=$(systemctl --user show "$SVC_NAME" -p MemoryPeak --value 2>/dev/null || echo 0)
    fi
    case "$_peak_bytes" in
        ''|*[!0-9]*) _peak_bytes=0 ;;
    esac
    if [ "$_peak_bytes" -gt 0 ] 2>/dev/null; then
        # peak(MiB, rounded up) + 73 MiB kernel charge + 474 MiB for evictable
        # file pages returning under pressure, then rounded up to a 16 MiB step
        # (512k pages; keeps the number tidy and never rounds a real peak DOWN).
        _peak_mib=$(( (_peak_bytes + 1048575) / 1048576 ))
        _ceiling=$(( _peak_mib + 73 + 474 ))
        _ceiling=$(( (_ceiling + 15) / 16 * 16 ))
        # Re-inject as a single authoritative line, immediately after the
        # OOMPolicy=stop directive where the committed file's memory block ends,
        # so a reader finds it next to the reasoning that produced it.
        _injected="$UNIT_DEST.tmp.$$"
        sed -e "s|^MemoryMax=.*||" -e "s|^# MemoryMax is generated.*||" \
            -e "s|^# peak (memory.peak).*||" -e "s|^# Delete this line.*||" \
            "$UNIT_DEST" | awk -v m="$_ceiling" '
            { print }
            /^OOMPolicy=stop$/ && !done {
                print "# MemoryMax is generated by install.sh from this machine MEASURED cgroup"
                print "# peak (memory.peak) + 73 MiB kernel charge + 474 MiB for evictable file pages."
                print "# Delete this line to fall back to no limit; see the block above for the recipe."
                printf "MemoryMax=%sM\n", m
                done = 1
            }' > "$_injected"
        mv "$_injected" "$UNIT_DEST"
        systemctl --user daemon-reload
        say "   MemoryMax=${_ceiling}M (measured peak ${_peak_mib} MiB + 73 MiB kernel + 474 MiB file cache)"
        say "   override it with: systemctl --user set-property $SVC_NAME MemoryMax=<size>"
    else
        say "   no MemoryMax set: no cgroup peak available yet (service not started)."
        say "   The unit runs unbounded. After it has served a request, set one with:"
        say "       systemctl --user set-property $SVC_NAME MemoryMax=<peak+512>M"
    fi
    if systemctl --user enable --now "$SVC_NAME" 2>/dev/null; then
        say "   systemctl --user enable --now $SVC_NAME: ok"
    else
        # A failing enable --now is expected when the model is not downloaded yet:
        # the unit's preflight sha256's the GGUF and exits 2, which Restart=on-failure
        # turns into a restart loop until the start limit trips. Saying so beats a
        # silent success that leaves the operator with a loop.
        say "   WARNING: the unit did not start (service start failed)."
        say "   Most likely the model is not downloaded yet -- see 'Fetch the model' below."
        say "   It is enabled and will start on next login; or start it after fetching:"
        say "       systemctl --user start $SVC_NAME"
        say "   Check the reason with:"
        say "       systemctl --user status $SVC_NAME"
        say "       journalctl --user -u $SVC_NAME -n 50 --no-pager"
    fi
else
    # No systemctl: say so, and give the exact manual steps. This is not an error:
    # the service runs perfectly well from service/run.sh.
    say "== systemctl not found -- skipping the systemd unit."
    say "   To install it manually, copy the unit and enable it:"
    say "       mkdir -p $HOME/.config/systemd/user"
    say "       cp $SRC/systemd/$UNIT_SRC $UNIT_DEST"
    say "       systemctl --user daemon-reload"
    say "       systemctl --user enable --now $SVC_NAME"
    say "   (On a machine without systemd, run $SRC/service/run.sh instead.)"
fi

cat <<EOF

== installed.

Extension is now discoverable by omp (package.json declares omp.extensions).
To verify:
    ls -l $EXT_DIR

The decision service (loopback only, port 61021):
    systemctl --user status $SVC_NAME      # if the unit was installed
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
    systemctl --user disable --now $SVC_NAME   # ignore the error if absent
    rm -f $UNIT_DEST
    systemctl --user daemon-reload
    rm $EXT_DIR
    rm -rf $INSTALL_DIR      # only if this installer cloned it
    # revert modelRoles.judge in ~/.omp/agent/config.yml if you changed it
EOF
