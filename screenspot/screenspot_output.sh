DATA_PATH="4o_v3_query.jsonl"
MODEL_PATH="../model/ferret-7b-v1.3"
ANSWER_FILE="ferret-7b-v1.3-org_v3_query.jsonl"

CUDA_VISIBLE_DEVICES=0 python -m screenspot.screenspot_output \
    --model-path $MODEL_PATH \
    --image_path screenspot_imgs \
    --data_path $DATA_PATH \
    --answers-file $ANSWER_FILE
