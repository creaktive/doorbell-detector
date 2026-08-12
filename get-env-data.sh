#!/usr/bin/env bash
curl -q -sL https://github.com/karoldvl/ESC-50/archive/master.zip | bsdtar -v -x -f - -C data/environment/ --strip-components 2 'ESC-50-master/audio/*.wav'
