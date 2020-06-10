#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -v
set -e


for path in $1/*; do
    $DIR/process_realsense_scan.sh $path
done

