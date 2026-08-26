#!/usr/bin/env bash
set -euo pipefail

# Set noise volume from the first argument; default to 0.9 if omitted
NOISE_VOL="${1:-0.9}"

for f in data/test/*.wav; do
    printf "%-40s\t" "$f"
    # sox "$f" -c 1 -r 16000 -e signed-integer -b 16 -t raw - \
    sox -m \
        "|sox \"$f\" -r 16000 -c 1 -p remix 1 gain -n -3" \
        "|sox -n -r 16000 -c 1 -p synth $(soxi -D "$f") brownnoise vol ${NOISE_VOL}" \
        -e signed-integer -b 16 -t raw - \
    | ./detect.py 2>/dev/null
done
