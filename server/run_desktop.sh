#!/bin/bash
# ------------------------------------------------------------------
# Bubble Tactile Sensor - Test & Recording Console (HDMI Desktop Launcher)
# Clean App Lifecycle: Closing the window automatically kills the Python backend!
# ------------------------------------------------------------------

# 1. Clean up any leftover dangling processes first
pkill -f "gui_optical_flow_dots.py" 2>/dev/null || true
pkill -9 -f "serverTest.py" 2>/dev/null || true
sleep 0.8

# 2. Start serverTest in background and capture its PID
cd /home/bubble/Desktop/TestVersion1
/usr/bin/python3 -u serverTest.py > server.log 2>&1 &
SERVER_PID=$!
echo "[Bubble] TestVersion1 Engine started with PID $SERVER_PID"

# Ensure cleanup on ANY exit (normal window close, Alt+F4, Ctrl+C, killed)
cleanup() {
    echo "[Bubble] App window closed. Cleanly terminating Python server..."
    kill -TERM $SERVER_PID 2>/dev/null || true
    sleep 0.4
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "serverTest.py" 2>/dev/null || true
    echo "[Bubble] Cleanly exited. Hardware released."
}
trap cleanup EXIT INT TERM

# 3. Wait for server to bind port 8080
for i in {1..25}; do
    if curl -s http://localhost:8080 > /dev/null 2>&1; then
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
if which chromium-browser > /dev/null; then
    chromium-browser \
        --ozone-platform-hint=auto \
        --app=http://localhost:8080 \
        --start-maximized \
        --no-first-run \
        --no-default-browser-check \
        --disable-translate \
        --user-data-dir="/tmp/chromium_bubble_test"
elif which chromium > /dev/null; then
    chromium \
        --ozone-platform-hint=auto \
        --app=http://localhost:8080 \
        --start-maximized \
        --no-first-run \
        --no-default-browser-check \
        --user-data-dir="/tmp/chromium_bubble_test"
fi

# When Chromium window closes, cleanup() runs automatically via trap!
