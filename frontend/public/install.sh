#!/bin/bash
# PCDS Agent Installer for macOS and Linux
# Usage: curl -sL https://pcdsai.app/install.sh | bash

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PCDS Agent Installer v1.0                       ║"
echo "║          Predictive Cyber Defense System                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    INSTALL_DIR="$HOME/Library/Application Support/PCDS"
    echo "🍎 Detected: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    INSTALL_DIR="/opt/pcds"
    echo "🐧 Detected: Linux"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.8 or later."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create install directory
echo "📁 Creating install directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Download agent
echo "📥 Downloading PCDS Agent..."
AGENT_URL="https://raw.githubusercontent.com/SANJAY31105/pcds-core/main/agent/pcds_tray_agent.py"
curl -sL "$AGENT_URL" -o "$INSTALL_DIR/pcds_agent.py"

# Create requirements
cat > "$INSTALL_DIR/requirements.txt" << EOF
psutil>=5.9.0
requests>=2.28.0
EOF

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -q -r "$INSTALL_DIR/requirements.txt" 2>/dev/null || pip install -q -r "$INSTALL_DIR/requirements.txt"

# Prompt for API key
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Enter your PCDS API Key (from https://pcdsai.app/download) ║"
echo "╚══════════════════════════════════════════════════════════════╝"
read -p "API Key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "⚠️  No API key provided. Using demo key."
    API_KEY="pcds_demo_key_12345"
fi

# Save config
cat > "$INSTALL_DIR/config.json" << EOF
{
    "api_key": "$API_KEY",
    "api_url": "https://pcds-backend-production.up.railway.app/api/v2/ingest",
    "poll_interval": 30
}
EOF

# Create launcher script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 pcds_agent.py
EOF
chmod +x "$INSTALL_DIR/start.sh"

# Create systemd service for Linux
if [[ "$OS" == "linux" ]]; then
    echo "🔧 Creating systemd service..."
    sudo cat > /etc/systemd/system/pcds-agent.service << EOF
[Unit]
Description=PCDS Security Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $INSTALL_DIR/pcds_agent.py
Restart=always
RestartSec=10
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PCDS_API_KEY=$API_KEY

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable pcds-agent
    sudo systemctl start pcds-agent
    echo "✓ PCDS Agent service started"
fi

# Create launchd plist for macOS
if [[ "$OS" == "macos" ]]; then
    echo "🔧 Creating launchd service..."
    cat > "$HOME/Library/LaunchAgents/com.pcds.agent.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pcds.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$INSTALL_DIR/pcds_agent.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PCDS_API_KEY</key>
        <string>$API_KEY</string>
    </dict>
</dict>
</plist>
EOF
    
    launchctl load "$HOME/Library/LaunchAgents/com.pcds.agent.plist"
    echo "✓ PCDS Agent service started"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ PCDS Agent installed successfully!                       ║"
echo "║  📊 View your data at: https://pcdsai.app/dashboard          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Commands:"
if [[ "$OS" == "linux" ]]; then
    echo "  Status: sudo systemctl status pcds-agent"
    echo "  Stop:   sudo systemctl stop pcds-agent"
    echo "  Logs:   sudo journalctl -u pcds-agent -f"
else
    echo "  Stop:   launchctl unload ~/Library/LaunchAgents/com.pcds.agent.plist"
    echo "  Start:  launchctl load ~/Library/LaunchAgents/com.pcds.agent.plist"
fi
