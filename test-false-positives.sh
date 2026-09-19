#!/usr/bin/env bash
# export DUMP_DETECTED=1
while IFS= read -r -d '' file; do
    ffmpeg -nostdin -hide_banner -loglevel error -i "$file" -ac 1 -ar 16000 -c:a pcm_s16le -f s16le pipe:1 | ./detect.py 2>/dev/null
done < <(find "$HOME" -name '.*' -prune -or -type f \( -name '*.ac3' -or -name '*.avi' -or -name '*.mkv' -or -name '*.mp3' \) -print0)
