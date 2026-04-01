#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== HTF Executor — Build macOS App ==="
echo ""

# 1. Activate venv
source venv/bin/activate

# 2. Build frontend
echo "[1/3] Building frontend..."
cd dashboard && npm run build && cd ..
echo "      Frontend built → dashboard/dist/"

# 3. Clean previous build
echo "[2/3] Cleaning previous build..."
rm -rf build dist

# 4. Build .app bundle
echo "[3/3] Building macOS app bundle..."
python setup_app.py py2app 2>&1 | tail -5

echo ""
echo "=== Done ==="
echo "App: dist/HTF Executor.app"
echo ""
echo "To install: drag to /Applications"
echo "To run: double-click the app"
