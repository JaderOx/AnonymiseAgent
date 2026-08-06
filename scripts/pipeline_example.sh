#!/bin/bash

for vc_method in pitch formant seedvc; do
    echo "=========================================="
    echo "开始 vc_method=${vc_method}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=? python agent.py \
        -i /path/to/data/ADReSS-IS2020-data/audio \
        -o /path/to/data/ADReSS-IS2020-data/workspace/anonymise/whisperx/output \
        --middle /path/to/data/ADReSS-IS2020-data/workspace/anonymise/whisperx/middle \
        --trans \
        --language 英文 \
        --spk_num 1 \
        --vc \
        --vc_method ${vc_method}
        --eval \
        --mode INFO

    if [ $? -ne 0 ]; then
        echo "vc_method=${vc_method} 运行失败，跳过继续"
    else
        echo "vc_method=${vc_method} 完成"
    fi

    echo ""
done

echo "全部 vc_method 跑完"

