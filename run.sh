#!/usr/bin/env bash
#
# Tennis Latency Arbitrage — Runner Script
#
# Usage:
#   ./run.sh                    # Single scan (dry run)
#   ./run.sh --loop             # Continuous scanning every 5 min
#   ./run.sh --loop --interval 3  # Every 3 min
#
# Deployment (cron example — run every 10 min during match hours):
#   */10 8-23 * * * cd /path/to/tennis_ML && ./run.sh >> logs/arb.log 2>&1
#
# DigitalOcean setup:
#   1. Create a $6/mo Droplet (Ubuntu 22.04)
#   2. Clone repo, install deps: pip install -r requirements.txt && playwright install chromium
#   3. Copy .env.example to .env, fill in API keys
#   4. Run: nohup ./run.sh --loop --interval 5 &
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Run the engine
exec python3 arbitrage_engine.py "$@"
