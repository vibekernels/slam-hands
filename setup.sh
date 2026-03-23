#!/bin/bash
# Build and install SVT-AV1 with AVX-512 support.
#
# The default Ubuntu/Debian libsvtav1 package is compiled with AVX2 only.
# This script rebuilds SVT-AV1 1.7.0 (same version/ABI) with AVX-512 enabled,
# which provides ~10% faster encoding on CPUs that support it (Zen 4/5, Ice Lake+).
#
# Requirements: cmake, nasm, build-essential
# Run with: sudo bash setup.sh

set -euo pipefail

# Check for AVX-512 support
if ! grep -q avx512 /proc/cpuinfo; then
    echo "CPU does not support AVX-512. Skipping SVT-AV1 rebuild."
    exit 0
fi

# Check current SVT-AV1 ASM level
CURRENT_ASM=$(ffmpeg -hide_banner -loglevel info -f lavfi -i "nullsrc=s=64x64:d=0.1,format=yuv420p" \
    -c:v libsvtav1 -preset 12 -g 2 -crf 30 -f null - 2>&1 | grep "asm level selected" || true)
if echo "$CURRENT_ASM" | grep -q "avx512"; then
    echo "SVT-AV1 already has AVX-512 enabled. Nothing to do."
    exit 0
fi

echo "Building SVT-AV1 1.7.0 with AVX-512 support..."

# Install build dependencies
apt-get install -y cmake nasm build-essential

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

cd "$WORKDIR"
git clone --depth 1 --branch v1.7.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=ON -DBUILD_SHARED_LIBS=ON
make -j"$(nproc)"

# Replace system library (same SONAME: libSvtAv1Enc.so.1)
SYSTEM_LIB=$(find /usr/lib -name "libSvtAv1Enc.so.1.7.0" 2>/dev/null | head -1)
if [ -z "$SYSTEM_LIB" ]; then
    echo "Error: system libSvtAv1Enc.so.1.7.0 not found"
    exit 1
fi

cp "$WORKDIR/SVT-AV1/Bin/Release/libSvtAv1Enc.so.1.7.0" "$SYSTEM_LIB"
ldconfig

# Verify
VERIFY=$(ffmpeg -hide_banner -loglevel info -f lavfi -i "nullsrc=s=64x64:d=0.1,format=yuv420p" \
    -c:v libsvtav1 -preset 12 -g 2 -crf 30 -f null - 2>&1 | grep "asm level selected")
echo "Installed. $VERIFY"
