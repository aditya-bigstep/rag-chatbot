#!/usr/bin/env bash
# One command to run ResearchPilot on a new machine:
#
#   ./start.sh            install what is missing, check the keys, start the app, open the browser
#   PORT=9000 ./start.sh  use another port
#   SKIP_CHECKS=1 ./start.sh   start without calling the APIs first
#
# Stop it with Ctrl+C.

set -e
cd "$(dirname "$0")"

say()  { printf "\n\033[1m%s\033[0m\n" "$1"; }
stop() { printf "\n\033[31m%s\033[0m\n\n" "$1"; exit 1; }

# 1. uv installs Python and every dependency for us.
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (the Python package manager this project uses)"
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || stop "uv was installed but is not on PATH. Open a new terminal and run ./start.sh again."
fi

# 2. Settings. A fresh checkout has no .env, and the app cannot run without at least one LLM key.
if [ ! -f .env ]; then
  cp .env.example .env
  stop "Created .env from .env.example. Open it, fill in GOOGLE_API_KEY or GROQ_API_KEY (and TAVILY_API_KEY, BREVO_API_KEY if you have them), then run ./start.sh again."
fi
missing=""
for name in LLM_PROVIDER LLM_MODEL EMBEDDING_MODEL SEARCH_PROVIDER; do
  grep -Eq "^${name}=.+" .env || missing="$missing $name"
done
[ -z "$missing" ] || stop "These settings are missing or empty in .env:$missing. Compare with .env.example."

# 3. Python packages (creates .venv the first time).
say "Installing dependencies"
uv sync --quiet

# 4. Are the keys and models actually working?
if [ -z "$SKIP_CHECKS" ]; then
  uv run python scripts/check_setup.py || exit 1
fi

# 5. Pick a free port, starting from PORT (default 8000).
port="${PORT:-8000}"
while lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
  echo "Port $port is in use, trying $((port + 1))"
  port=$((port + 1))
done
url="http://localhost:$port"

# 6. Start the server, wait until it answers, then open the browser.
say "Starting ResearchPilot on $url"
uv run uvicorn app.main:app --port "$port" &
server=$!
trap 'kill $server 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
  if curl -s -o /dev/null "$url/health"; then
    ready=1
    break
  fi
  kill -0 $server 2>/dev/null || stop "The server stopped while starting. Read the error above."
  sleep 1
done
[ -n "$ready" ] || stop "The server did not answer within 60 seconds."

if command -v open >/dev/null 2>&1; then
  open "$url"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 &
fi
say "Running at $url   (Ctrl+C to stop)"
wait $server
