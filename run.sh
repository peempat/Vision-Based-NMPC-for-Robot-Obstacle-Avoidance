#!/usr/bin/env bash
# Wrapper that injects extracted X11 libs so cv2's Qt XCB plugin can load.
# Usage:  ./run.sh [args...]   e.g.  ./run.sh --mode cv_test
export LD_LIBRARY_PATH=/tmp/xlibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
source "$(dirname "$0")/../myenv/bin/activate"
python "$(dirname "$0")/main.py" "$@"
