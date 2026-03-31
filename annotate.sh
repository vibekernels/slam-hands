#!/usr/bin/env bash
# Annotate a video with the recommended settings.
# Usage: ./annotate.sh /path/to/video.mov [output_dir]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_file> [output_dir]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-}"

ARGS=(
    "$INPUT"
    --fast-traj
    --hand-stride 1
)

if [ -n "$OUTPUT" ]; then
    ARGS+=(-o "$OUTPUT")
fi

exec python3 "$(dirname "$0")/annotate_pipeline.py" "${ARGS[@]}"
