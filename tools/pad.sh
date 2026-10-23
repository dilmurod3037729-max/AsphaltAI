#!/bin/bash
# Padga skript yuboradi: bash tools/pad.sh "steer 0.5; gas 1; wait 300; shockwave; stop"
export MSYS_NO_PATHCONV=1
/c/platform-tools/adb.exe -s emulator-5554 shell "am broadcast -a space.asphaltai.PAD --es cmd '$*'" > /dev/null
