#!/usr/bin/env bash
# Augment .wav files in data/{downstairs,upstairs} using sox, and download ESC-50 dataset to data/environment.
# Output of augmentation: same directory as source with -aug-<transform>.wav suffix.

set -euo pipefail

# Augmentation definitions: suffix sox_args...
AUGMENTS=(
    # Speed & Pitch
    "-aug-speed-0-9 speed 0.9"
    "-aug-speed-1-1 speed 1.1"
    "-aug-tempo-0-95 tempo 0.95"
    "-aug-tempo-1-05 tempo 1.05"
    "-aug-pitch-200 pitch -200"
    "-aug-pitch+200 pitch +200"

    # Volume & Dynamics
    "-aug-vol-0-7 vol 0.7"
    "-aug-vol-1-3 vol 1.3"
    "-aug-overdrive overdrive 10 10"
    "-aug-compand compand 0.3,1 6:-70,-60,-20 -5 -90 0.2"

    # Filtering & EQ
    "-aug-lowpass lowpass 2000"
    "-aug-highpass highpass 300"
    "-aug-phone bandpass 1700 2700"
    "-aug-eq-dip equalizer 1000 1q -6"
    "-aug-proximity bass +6"

    # Space, Delay & Alignment
    "-aug-reverb reverb 50 50 50"
    "-aug-echo echo 0.8 0.88 60 0.4"
    "-aug-flanger flanger 0 3 0 71 0.5 25 linear"
    "-aug-chorus chorus 0.7 0.8 45 0.35 0.25 1.5 -t"
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
done < <(find 'data/downstairs/' 'data/upstairs/' -type f -name '*.wav' -print0)

echo "Done: $count augmented, $fail failed."

curl -q -sL https://github.com/karoldvl/ESC-50/archive/master.zip \
    | bsdtar -v -x -f - -C 'data/environment/' --strip-components 2 'ESC-50-master/audio/*.wav'
