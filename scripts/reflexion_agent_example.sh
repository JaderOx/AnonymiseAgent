#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir=${1:?Usage: reflexion_agent_example.sh INPUT_DIR OUTPUT_DIR [REQUEST]}
output_dir=${2:?Missing OUTPUT_DIR}
request=${3:-语言为中文，说话人数量为1，医疗任务为抑郁检测。不需要内容匿名，希望保留韵律。}
python reflexion_agent_downstream.py -i "$input_dir" -o "$output_dir" \
    --middle_dir "$output_dir/middle" -r "$request"
