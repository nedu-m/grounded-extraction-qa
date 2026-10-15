#!/usr/bin/env bash
#
# Deploy grounded-extraction-qa to the application host.
#
# Run from the repository root on a machine that can reach the host over SSH.
#
#   ./deploy/deploy.sh                     # code only; the key is preserved
#   DEEPSEEK_API_KEY=sk-... ./deploy/deploy.sh   # set or rotate the key too
#   HOST=other-box ./deploy/deploy.sh
#
# The key is never printed, never written to the repository, and never passed as
# a command-line argument where it would land in the host's process list and
# shell history. It travels over stdin and lands in a 0600 file.
#
# What this does NOT do: create the DNS record. That is a one-off, and doing it
# from here would mean holding a zone-wide Cloudflare token on an application
# host for a change that happens twice a year.

set -euo pipefail

HOST="${HOST:-ikenga}"
REMOTE_DIR="${REMOTE_DIR:-/opt/grounded-extraction-qa}"
PORT="${PORT:-8099}"
DOMAIN="${DOMAIN:-qa.nedum.xyz}"
SERVICE="geq"
CADDY_FILE="/etc/caddy/conf.d/${DOMAIN}.caddy"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say() { printf '  %s\n' "$*"; }
fail() { printf '!! %s\n' "$*" >&2; exit 1; }

echo "==> checking the host is reachable"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" true || fail "cannot reach $HOST over SSH"

echo "==> ensuring $REMOTE_DIR exists and is owned by the deploy user"
# /opt is root-owned. The directory is created once with sudo and then owned by the
# deploy user, so every later sync needs no privilege.
ssh "$HOST" bash -euo pipefail <<REMOTE
  if [ ! -d '$REMOTE_DIR' ]; then
    sudo install -d -o "\$(id -un)" -g "\$(id -gn)" -m 0755 '$REMOTE_DIR'
    echo "  created $REMOTE_DIR"
  fi
REMOTE

echo "==> syncing the repository to $HOST:$REMOTE_DIR"
# .env is excluded deliberately: it holds the key and is written separately below.
# reports/ is generated output; the host regenerates it.
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.env' \
  --exclude 'reports/*.json' \
  --exclude 'reports/*.md' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.github/' \
  --exclude 'node_modules/' \
  ./ "$HOST:$REMOTE_DIR/"

if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  echo "==> writing the environment file (key supplied)"
  # umask 077 before the redirect so the file is 0600 from the moment it exists,
  # and printf rather than echo so the key is not interpreted.
  printf 'umask 077\nDEEPSEEK_API_KEY=%s\nGEQ_LLM_MODEL=%s\n' \
    "$DEEPSEEK_API_KEY" "${GEQ_LLM_MODEL:-deepseek-chat}" \
    | ssh "$HOST" "cat > '$REMOTE_DIR/.env.tmp' && mv '$REMOTE_DIR/.env.tmp' '$REMOTE_DIR/.env'"
  say "wrote $REMOTE_DIR/.env as 0600"
else
  echo "==> no key supplied: leaving the existing environment file in place"
  ssh "$HOST" "test -f '$REMOTE_DIR/.env' || printf 'umask 077\n# Set DEEPSEEK_API_KEY to enable the live-model extractor.\n' > '$REMOTE_DIR/.env'"
fi

echo "==> creating the virtualenv and installing"
# uv is not installed on the host, and pip into a system Python is not something
# to do on a shared box, so this is a plain venv.
ssh "$HOST" bash -euo pipefail <<REMOTE
  cd '$REMOTE_DIR'
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
  fi
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -e .
  .venv/bin/python -c "import geq; print('  installed geq', geq.__version__)"
REMOTE

echo "==> building the scorecards on the host (this is the CI gate, on real data)"
ssh "$HOST" bash -euo pipefail <<REMOTE
  cd '$REMOTE_DIR'
  for extractor in rules:v1 rules:v2 rules:v3 adversarial:uncited adversarial:fabricating; do
    .venv/bin/geq run --extractor "\$extractor" --out "reports/\${extractor//:/-}" >/dev/null
  done
  .venv/bin/geq compare --baseline reports/rules-v2.json --candidate reports/rules-v3.json >/dev/null
  echo "  regression gate passed on the host"
REMOTE

echo "==> installing the systemd unit"
scp -q "deploy/${SERVICE}.service" "$HOST:/tmp/${SERVICE}.service"
ssh "$HOST" bash -euo pipefail <<REMOTE
  sudo install -m 0644 /tmp/${SERVICE}.service /etc/systemd/system/${SERVICE}.service
  rm -f /tmp/${SERVICE}.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now ${SERVICE}.service >/dev/null
  sudo systemctl restart ${SERVICE}.service
  sleep 2
  systemctl is-active --quiet ${SERVICE}.service || {
    journalctl -u ${SERVICE}.service -n 40 --no-pager
    exit 1
  }
REMOTE
say "service active on 127.0.0.1:${PORT}"

echo "==> installing the Caddy site block"
scp -q "deploy/${DOMAIN}.caddy" "$HOST:/tmp/${DOMAIN}.caddy"
ssh "$HOST" bash -euo pipefail <<REMOTE
  sudo install -m 0644 /tmp/${DOMAIN}.caddy '$CADDY_FILE'
  rm -f /tmp/${DOMAIN}.caddy
  # Validate before reloading. A malformed file here would take down every other
  # site on this host, not just this one.
  sudo caddy validate --config /etc/caddy/Caddyfile >/dev/null
  sudo systemctl reload caddy
  systemctl is-active --quiet caddy || { echo '!! caddy is not running'; exit 1; }
REMOTE

echo "==> other sites on this host must still be up"
# Beak's deploy overwrites the shared Caddyfile wholesale, and Caddy keeps
# serving happily from a config that has lost a site. Checking a hostname that is
# not ours is the only way to catch that a reload broke someone else.
for probe in https://meetbeak.com/ https://jiayou.nedum.xyz/; do
  # A redirect is a healthy answer: meetbeak.com has a www block that 301s to the
  # apex by design. Anything that is not a 3xx or a 2xx is the failure signal.
  code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$probe" || true)
  printf '  %-32s -> HTTP %s\n' "$probe" "$code"
  case "$code" in
    2*|3*) : ;;
    *) fail "$probe is not answering (HTTP $code); a reload may have broken it" ;;
  esac
done

echo "==> health check through the loopback listener"
ssh "$HOST" "curl -fsS http://127.0.0.1:${PORT}/health" | head -c 400
echo

echo "==> health check through the public hostname"
# `caddy validate` passes on a config that is missing this site entirely, which
# is how an outage got through on this host before. Ask the public name.
#
# SKIP_PUBLIC_CHECK=1 exists for the first deploy, before the DNS record is in
# place. It is deliberately not the default: a deploy that cannot show the site
# answering is not finished, and the whole point of checking here is that Caddy
# will happily serve a config that is missing this site.
if [ -n "${SKIP_PUBLIC_CHECK:-}" ]; then
  say "skipped (SKIP_PUBLIC_CHECK is set); run again without it once DNS exists"
elif curl -s -o /dev/null -m 25 -w '%{http_code}' "https://${DOMAIN}/" | grep -q '^200$'; then
  say "https://${DOMAIN}/ -> HTTP 200"
else
  cat <<NOTE
!! https://${DOMAIN}/ is not answering.

   Most likely cause, if this is the first deploy: the DNS record does not exist
   yet. Add a grey-cloud A record:

       ${DOMAIN%%.*}   A   84.247.142.249   (DNS only, not proxied)

   Then re-run this script. Everything else is already in place.

   If the record does exist, note that a tailnet-joined machine may resolve the
   name to a private address and time out while the public path is fine. Test the
   real path with:
       curl --resolve ${DOMAIN}:443:\$(dig +short ${DOMAIN} @1.1.1.1 | head -1) https://${DOMAIN}/
NOTE
  exit 1
fi

echo "==> done"
say "https://${DOMAIN}/"
