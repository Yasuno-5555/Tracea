#!/bin/bash
echo "Tracea Metal Verification"
echo "========================="

if ! command -v cargo &> /dev/null; then
    echo "❌ 'cargo' not found in PATH."
    echo "Please ensure Rust is installed and 'cargo' is available."
    echo "Typical fix: source \$HOME/.cargo/env"
    exit 1
fi

echo "Running Metal Smoke Test..."
cargo test --features metal --test metal_basic -- --nocapture

if [ $? -eq 0 ]; then
    echo "✅ Metal Backend Verified Successfully!"
else
    echo "❌ Verification Failed."
    exit 1
fi
