#!/bin/bash
# ------------------------------------------------------------------
# Bubble Tactile Sensor - Test & Recording Console (HDMI Desktop Launcher)
# Clean App Lifecycle: Closing the window automatically kills the Python backend!
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8080}"

# 1. Clean up any leftover dangling processes first
pkill -9 -f "server/server.py" 2>/dev/null || true
sleep 0.8

# 2. Start server in background and capture its PID
cd "$REPO_ROOT" || exit 1
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -u server/server.py > server.log 2>&1 &
SERVER_PID=$!
echo "[Bubble] Server started with PID $SERVER_PID (cwd: $REPO_ROOT)"

# Ensure cleanup on ANY exit (normal window close, Alt+F4, Ctrl+C, killed)
cleanup() {
    echo "[Bubble] App window closed. Cleanly terminating Python server..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 0.4
    kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -9 -f "server/server.py" 2>/dev/null || true
    echo "[Bubble] Cleanly exited. Hardware released."
}
trap cleanup EXIT INT TERM

# 3. Wait for server to bind port
for _ in {1..25}; do
    if curl -s "http://localhost:${PORT}" > /dev/null 2>&1; then
        echo "[Bubble] Server is ready."
        break
    fi
    sleep 0.3
done

# 4. Ensure Wayland / X11 environment
if [ -z "$WAYLAND_DISPLAY" ] && [ -S "/run/user/$(id -u)/wayland-0" ]; then
    export WAYLAND_DISPLAY=wayland-0
fi
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# 5. Open Dedicated App Window in FOREGROUND (Blocks until user closes window!)
if command -v chromium-browser > /dev/null 2>&1; then
    chromium-browser \
        --ozone-platform-hint=auto \
        --app="http://localhost:${PORT}" \
        --start-maximized \
        --no-first-run \
        --no-default-browser-check \
        --disable-translate \
        --user-data-dir="/tmp/chromium_bubble_test"
elif command -v chromium > /dev/null 2>&1; then
    chromium \
        --ozone-platform-hint=auto \
        --app="http://localhost:${PORT}" \
        --start-maximized \
        --no-first-run \
        --no-default-browser-check \
        --disable-translate \
        --user-data-dir="/tmp/chromium_bubble_test"
fi

# When Chromium window closes, cleanup() runs automatically via trap!
