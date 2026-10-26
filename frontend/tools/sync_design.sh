#!/bin/sh
# Regenerate the views from the Claude Design files in design/ (run after pulling a new design).
# design/BloomBot.dc.html carries a few extra {{ bindings }} for live data; keep them when you
# replace it with a newer export (git diff design/ shows them).
set -e
cd "$(dirname "$0")/.."
python3 tools/dc2jsx.py design/BloomBot.dc.html src/views/AppView.jsx --imports "import DistBar from '../components/DistBar.jsx';\nimport LevelLadder from '../components/LevelLadder.jsx';\nimport RewriteCard from '../components/RewriteCard.jsx';"
for c in DistBar LevelLadder RewriteCard; do
  python3 tools/dc2jsx.py design/$c.dc.html src/components/$c.jsx --component $c
done
{ printf "// GENERATED from design/bloombot-data.js — the design's static content and recorded runs.\n/* eslint-disable */\nconst window = {};\n"
  cat design/bloombot-data.js
  printf "\nexport const BB = window.BB;\n"; } > src/designData.js
echo "wrote src/designData.js"
