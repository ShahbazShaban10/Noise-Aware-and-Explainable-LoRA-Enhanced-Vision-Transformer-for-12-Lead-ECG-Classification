#!/usr/bin/env bash
# Container entry point for the repository's own docker-compose.yml.
#
# It exists for one reason: the environment image deliberately does NOT contain the
# reference package. environment/Dockerfile must never bake solution/ or tests/ in, because
# that would hand the reference implementation to every agent Harbor runs -- including the
# do-nothing agent, which would then score above zero. Under Harbor that is fine, because
# Harbor uploads solution/ at run time and only for the oracle agent, and solve.sh installs
# it as its first step.
#
# But `docker compose run --rm verify` on a fresh checkout never runs solve.sh, so nothing
# had ever installed the package and test.sh failed its own precondition:
#
#   FAIL: no importable package named 'ecgvit' -- nothing was built
#
# which reads like a broken submission and is really just a missing install. This script
# installs the package from the mounted /solution when it is not already importable, so
# verify grades artefacts and the unit tier runs standalone. The image on disk is untouched:
# the install happens in the container's writable layer, which `--rm` discards.
#
# Set ECGVIT_AUTO_INSTALL=0 to skip it -- the solve service does, because solve.sh performs
# its own install and doing it twice is pure waste.

set -euo pipefail

SOLUTION_DIR="${SOLUTION_DIR:-/solution}"

if [ "${ECGVIT_AUTO_INSTALL:-1}" = "1" ] && ! python -c "import ecgvit" 2>/dev/null; then
  if [ -f "$SOLUTION_DIR/pyproject.toml" ]; then
    printf '\n\033[1m[entrypoint]\033[0m installing ecgvit from %s\n' "$SOLUTION_DIR"

    # Build from a writable copy, never from $SOLUTION_DIR itself. docker-compose.yml mounts
    # it :ro, and with --no-build-isolation the setuptools backend writes build/lib/ inside
    # the source tree -- "could not create 'build/lib/ecgvit': Read-only file system".
    # solve.sh copies for exactly the same reason; see the comment there.
    BUILD_TMP="$(mktemp -d)"
    trap 'rm -rf "$BUILD_TMP"' EXIT
    cp -r "$SOLUTION_DIR" "$BUILD_TMP/ecgvit-src"
    python -m pip install --quiet --no-deps --no-build-isolation "$BUILD_TMP/ecgvit-src"
    python -c "import ecgvit, pathlib; print('  ecgvit at', pathlib.Path(ecgvit.__file__).parent)"
  else
    # Not fatal here. test.sh and solve.sh both check for the package themselves and report
    # it far better than this script could; failing now would only hide their message.
    printf '[entrypoint] no pyproject.toml under %s; skipping install\n' "$SOLUTION_DIR" >&2
  fi
fi

exec "$@"
