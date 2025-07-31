BASE_PATH=${1}
CKPT_NAME=${2-"gpt2-base"}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/../data/dolly/ \
    --processed-data-dir ${BASE_PATH}/../data/processed_data/dolly/prompt/ \
    --model-path ${BASE_PATH}/../checkpoints/${CKPT_NAME} \
    --data-process-workers 16 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type ${CKPT_NAME}

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/../data/dolly/ \
    --processed-data-dir ${BASE_PATH}/../data/processed_data/dolly/full/ \
    --model-path ${BASE_PATH}/../checkpoints/${CKPT_NAME} \
    --data-process-workers 16 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type ${CKPT_NAME}
