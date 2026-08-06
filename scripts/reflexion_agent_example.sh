CUDA_VISIBLE_DEVICES=6 python reflexion_agent_downstream.py \
    -i /path/to/Anonymise_Agent/demo/workspace/user_input/audio \
    -o /path/to/Anonymise_Agent/demo/workspace/output \
    --middle_dir /path/to/Anonymise_Agent/demo/workspace/middle \
    --mode INFO \
    --llm Qwen3.5-9B \
    -r 语言为中文，说话人数量为1，医疗任务为抑郁检测。我希望mos能比原始音频更高