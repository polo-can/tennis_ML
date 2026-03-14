#!/usr/bin/env bash
#
# Server setup script for Tennis Arbitrage Engine
# Run this ON the DigitalOcean droplet after SSHing in as root
#
set -euo pipefail

echo "=== Tennis Arbitrage Engine — Server Setup ==="

# 1. System packages
echo "[1/7] Installing system packages..."
apt update -qq
apt install -y -qq python3 python3-pip python3-venv git curl

# 2. Create dedicated user
echo "[2/7] Creating tennis user..."
if ! id -u tennis &>/dev/null; then
    adduser --disabled-password --gecos "" tennis
fi

# 3. Clone repo
echo "[3/7] Cloning repository..."
sudo -u tennis bash -c '
    cd /home/tennis
    if [ ! -d tennis_ML ]; then
        git clone https://github.com/YOUR_USERNAME/tennis_ML.git
    else
        cd tennis_ML && git pull
    fi
'

# 4. Python dependencies
echo "[4/7] Installing Python dependencies..."
sudo -u tennis bash -c '
    cd /home/tennis/tennis_ML
    pip3 install --user -r requirements.txt
'

# 5. Playwright + Chromium
echo "[5/7] Installing Playwright & Chromium..."
sudo -u tennis bash -c '
    pip3 install --user playwright
    python3 -m playwright install --with-deps chromium
'

# 6. Create data directory
echo "[6/7] Setting up directories..."
sudo -u tennis mkdir -p /home/tennis/tennis_ML/data

# 7. Install systemd service
echo "[7/7] Installing systemd service..."
cp /home/tennis/tennis_ML/deploy/tennis-arb.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable tennis-arb

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Create your .env file:"
echo "     sudo -u tennis nano /home/tennis/tennis_ML/.env"
echo ""
echo "  2. Start the service:"
echo "     systemctl start tennis-arb"
echo ""
echo "  3. Check it's running:"
echo "     systemctl status tennis-arb"
echo "     journalctl -u tennis-arb -f"
