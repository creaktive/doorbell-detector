#!/usr/bin/env bash
# Augment all .wav files in data/ recursively using sox.
# Output: same directory as source with -aug-<transform>.wav suffix.

set -euo pipefail

SOURCES_DIR="${1:-data}"

if [ ! -d "$SOURCES_DIR" ]; then
    echo "Error: '$SOURCES_DIR' is not a directory." >&2
    exit 1
fi

# Augmentation definitions: suffix sox_args...
AUGMENTS=(
    "-aug-speed-0-9 speed 0.9"
    "-aug-speed-1-1 speed 1.1"
    "-aug-pitch-200 pitch -200"
    "-aug-pitch+200 pitch +200"
    "-aug-vol-0-7 vol 0.7"
    "-aug-vol-1-3 vol 1.3"
    "-aug-reverb reverb 50 50 50"
    "-aug-lowpass lowpass 2000"
    "-aug-highpass highpass 300"
)

count=0
fail=0

while IFS= read -r -d '' wav; do
    base="${wav%.wav}"
    for aug in "${AUGMENTS[@]}"; do
        suffix="${aug%% *}"
        args="${aug#* }"
        out="${base}${suffix}.wav"
        if sox "$wav" "$out" $args 2>/dev/null; then
            count=$((count + 1))
        else
            echo "FAIL: $wav -> $out" >&2
            fail=$((fail + 1))
        fi
    done
done < <(find "$SOURCES_DIR" -type f -name '*.wav' -print0)

echo "Done: $count augmented, $fail failed."
