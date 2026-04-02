#!/bin/bash
# build_app.sh — Builds HTF Executor as a distributable macOS .app + .dmg
set -e

cd "$(dirname "$0")"

echo "═══════════════════════════════════════"
echo "  HTF Executor — App Bundle Builder"
echo "═══════════════════════════════════════"

# 1. Build the React dashboard
echo ""
echo "→ [1/3] Building dashboard..."
cd dashboard && npm run build && cd ..
echo "   ✓ Dashboard built"

# 2. Activate venv and run PyInstaller
echo ""
echo "→ [2/3] Bundling with PyInstaller..."
source venv/bin/activate
pip install pyinstaller --quiet

# Clean old build artifacts
rm -rf build dist/HTFExecutor dist/HTFExecutor.app

pyinstaller HTFExecutor.spec --clean --noconfirm
echo "   ✓ App bundle created: dist/HTFExecutor.app"

# 3. Package as DMG
echo ""
echo "→ [3/3] Packaging as DMG..."
rm -f "dist/HTFExecutor.dmg"

hdiutil create -volname "HTFExecutor" \
    -srcfolder "dist/HTFExecutor.app" \
    -ov -format UDZO \
    "dist/HTFExecutor.dmg"
echo "   ✓ DMG created: dist/HTFExecutor.dmg"

echo ""
echo "═══════════════════════════════════════"
echo "  Done!"
echo "  App:  dist/HTFExecutor.app"
echo "  DMG:  dist/HTFExecutor.dmg"
echo ""
echo "  Share dist/HTFExecutor.dmg"
echo "  Config/logs: ~/Library/Application Support/HTFExecutor/"
echo "═══════════════════════════════════════"
