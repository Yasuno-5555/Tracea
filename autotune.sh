#!/bin/bash
set -e

echo "Starting Autotuning..."

# Synthetic ResNet Block (run_resnet_block)
echo "Tuning Synthetic Block (32ch)..."
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo run --bin tuner --release --no-default-features -- conv 1 64 64 32 32 3 3 1 1

# Standard ResNet-18 Shapes
echo "Tuning ResNet-18 Layers..."
# Stem
cargo run --bin tuner --release --no-default-features  -- conv 1 224 224 3 64 7 7 4 3

# Layer 1
cargo run --bin tuner --release --no-default-features  -- conv 1 56 56 64 64 3 3 1 1

# Layer 2
cargo run --bin tuner --release --no-default-features  -- conv 1 56 56 64 128 3 3 2 1
cargo run --bin tuner --release --no-default-features  -- conv 1 56 56 64 128 1 1 2 0
cargo run --bin tuner --release --no-default-features  -- conv 1 28 28 128 128 3 3 1 1

# Layer 3
cargo run --bin tuner --release --no-default-features  -- conv 1 28 28 128 256 3 3 2 1
cargo run --bin tuner --release --no-default-features  -- conv 1 28 28 128 256 1 1 2 0
cargo run --bin tuner --release --no-default-features  -- conv 1 14 14 256 256 3 3 1 1

# Layer 4
cargo run --bin tuner --release --no-default-features  -- conv 1 14 14 256 512 3 3 2 1
cargo run --bin tuner --release --no-default-features  -- conv 1 14 14 256 512 1 1 2 0
cargo run --bin tuner --release --no-default-features  -- conv 1 7 7 512 512 3 3 1 1

echo "Autotuning Complete!"
