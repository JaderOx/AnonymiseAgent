CUDA_VISIBLE_DEVICES=4 \
python anonymise_infer.py \
    --save_path /path/to/anonymise/crisis/seedvc_fixspk_wo_text \
    --source_path /path/to/crisis/raw_data_16khz \
    --diarization_path /path/to/data/crisis/Exp1/middle/trans \
    2>&1 | tee -a "infer_log.log"


# --source_path /path/to/crisis/raw_data_16khz \
# --source_path /path/to/data/crisis/Exp1/output/origin \

# CUDA_VISIBLE_DEVICES=5 \
# python anonymise_infer.py \
#     --save_path output \
#     --source_path 201901038290.wav \
#     --diarization_path /path/to/data/crisis/Exp1/middle/trans \
#     2>&1 | tee -a "infer_log.log"