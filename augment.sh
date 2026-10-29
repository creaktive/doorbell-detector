#!/usr/bin/env bash
# Augment .wav files in data/{downstairs,upstairs} using sox, download ESC-50 dataset to data/environment,
# and write pure sine tones around 1 kHz there as negatives.
# Output of augmentation: same directory as source with -aug-<transform>.wav suffix.

set -euo pipefail

# Augmentation definitions: suffix sox_args...
AUGMENTS=(
    # No speed, tempo or pitch shifts: they teach "any tone near 1 kHz is a bell" and raise false positives
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

augment_files() {
    while IFS= read -r -d '' wav; do
        local base="${wav%.wav}"
        for aug in "${AUGMENTS[@]}"; do
            local suffix="${aug%% *}"
            local args="${aug#* }"
            local out="${base}${suffix}.wav"
            # -R: fixed dither seed, so regenerated files are byte-identical
            if sox -R "$wav" "$out" $args 2>/dev/null; then
                count=$((count + 1))
            else
                echo "FAIL: $wav -> $out" >&2
                fail=$((fail + 1))
            fi
        done
    done
    echo "Done: $count augmented, $fail failed."
}

if [ $# -gt 0 ]; then
    augment_files < <(for wav in "$@"; do printf '%s\0' "$wav"; done)
else
    augment_files < <(find 'data/downstairs/' 'data/upstairs/' -type f -name '*.wav' -print0)

    curl -q -sL https://github.com/karoldvl/ESC-50/archive/master.zip \
        | bsdtar -v -x -f - -C 'data/environment/' --strip-components 2 'ESC-50-master/audio/*.wav'

    # The upstairs bell is 976 Hz plus odd harmonics, and the model fired UPSTAIRS on a bare 1 kHz sine.
    for f in $(seq 850 10 1150); do
        sox -R -n -r 16000 -c 1 -b 16 "data/environment/synthetic-sine-$(printf '%04d' "$f").wav" synth 5 sine "$f" vol 0.5
    done
fi
