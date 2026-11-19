CUDA_VISIBLE_DEVICES=6
python casa.py \
    -i examples/audio \
    -o examples/audio_out \
    --middle_dir examples/audio_middle \
    -r "这份数据是中文录音，单个说话人，对这份数据做匿名化，要求CER尽量低，同时EER大于30%。需要做声纹匿名和文本匿名。" \