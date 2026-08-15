#!/usr/bin/env bash

for f in data/test/*.wav; do
    printf "%-40s\t" "$f"
    sox "$f" -c 1 -r 16000 -e signed-integer -b 16 -t raw - | ./detect.py 2>/dev/null
done
