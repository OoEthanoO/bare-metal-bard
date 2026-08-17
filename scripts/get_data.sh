#!/usr/bin/env bash
# Fetch TinyShakespeare (~1.1 MB). The dataset is not committed; it is a
# well-known public file and downloading keeps the repo small.
set -euo pipefail

mkdir -p data
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if [ -f data/input.txt ]; then
  echo "data/input.txt already present ($(wc -c < data/input.txt) bytes)"
  exit 0
fi

curl -sSL --max-time 120 -o data/input.txt "$URL"
echo "downloaded data/input.txt ($(wc -c < data/input.txt) bytes)"
