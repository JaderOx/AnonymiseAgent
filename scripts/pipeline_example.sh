#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir=${1:?Usage: pipeline_example.sh INPUT_DIR OUTPUT_DIR [LANGUAGE]}
output_dir=${2:?Missing OUTPUT_DIR}
language=${3:-中文}
python agent.py -i "$input_dir" -o "$output_dir" --middle_dir "$output_dir/middle" \
    --trans --language "$language" --spk_num 1 --vc --vc_method NeuralVC --eval
