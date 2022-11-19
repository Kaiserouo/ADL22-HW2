
# whether to run those operation, set to 1 if want to run
DO_PREPROCESS_DATASET=1
RUN_SWAG=1
RUN_SWAG_POSTPROCESS=1
RUN_QA=1
RUN_QA_POSTPROCESS=1

# arguments, default as follows:
# CONTEXT_PATH='./data/context.json'
# TEST_PATH='./data/test.json'
# OUTPUT_CSV_PATH='./predict.csv'
CONTEXT_PATH="${1}"
TEST_PATH="${2}"
OUTPUT_CSV_PATH="${3}"

# CS and QA model directory
SWAG_DIR="./output_model/swag"
QA_DIR="./output_model/qa"

# directory for storing preprocessed data
# SHOULD CONTAIN 4 FILES: {swag,squad}_{test,valid}.json (ref. README.md)
PREPROCESS_DATASET_DIR="./output_model/preprocess_dataset"

# ---

echo '[*] Generating Directories'   # just to make sure...
mkdir -p "${PREPROCESS_DATASET_DIR}"

# ---

[[ DO_PREPROCESS_DATASET -eq '1' ]] && \
    echo '[*] Executing preprocess_dataset' && \
    python3.9 preprocess_dataset.py \
        --only_process_test_file \
        --context_path "${CONTEXT_PATH}" \
        --test_path "${TEST_PATH}" \
        --output_data_dir "${PREPROCESS_DATASET_DIR}"
# ---

cs_model_path="${SWAG_DIR}/pytorch_model.bin"
config_name="${SWAG_DIR}/config.json"
tokenizer_name="${SWAG_DIR}"

train_file="${PREPROCESS_DATASET_DIR}/swag_train.json"
valid_file="${PREPROCESS_DATASET_DIR}/swag_valid.json"
test_file="${PREPROCESS_DATASET_DIR}/swag_test.json"
test_cs_output="${PREPROCESS_DATASET_DIR}/test_cs_output.json"

max_len=512
batch_size=1

[[ RUN_SWAG -eq '1' ]] && \
    echo '[*] Executing run_swag testing' && \
    python3.9 run_swag_no_trainer.py \
        --model_name_or_path "${cs_model_path}" \
        --config_name "${config_name}" \
        --tokenizer_name "${tokenizer_name}" \
        --train_file "${train_file}" \
        --validation_file "${valid_file}" \
        --test_file "${test_file}" \
        --test_output_file "${test_cs_output}" \
        --per_device_eval_batch_size "${batch_size}" \
        --max_length "${max_len}" \
        --num_train_epochs 0 \
        --no_metric

# ----------

test_squad_file="${PREPROCESS_DATASET_DIR}/squad_test.json"

[[ RUN_SWAG_POSTPROCESS -eq '1' ]] && \
    echo '[*] Executing postprocessing CS output' && \
    python3.9 process_cs_output.py \
        --context_file "${CONTEXT_PATH}" \
        --test_file "${TEST_PATH}" \
        --input_cs_file "${test_cs_output}" \
        --output_squad_file "${test_squad_file}"

# ----------

qa_model_path="${QA_DIR}/pytorch_model.bin"
config_name="${QA_DIR}/config.json"
tokenizer_name="${QA_DIR}"

train_file="${PREPROCESS_DATASET_DIR}/squad_train.json"
valid_file="${PREPROCESS_DATASET_DIR}/squad_valid.json"
test_file="${PREPROCESS_DATASET_DIR}/squad_test.json"
test_squad_output_dir="${PREPROCESS_DATASET_DIR}"

batch_size=1

[[ RUN_QA -eq '1' ]] && \
    echo '[*] Executing run_qa testing' && \
    python3.9 run_qa_no_trainer.py \
        --model_name_or_path "${qa_model_path}" \
        --config_name "${config_name}" \
        --tokenizer_name "${tokenizer_name}" \
        --train_file "${train_file}" \
        --validation_file "${valid_file}" \
        --do_predict \
        --test_file "${test_file}" \
        --test_output_dir "${test_squad_output_dir}" \
        --per_device_eval_batch_size "${batch_size}" \
        --max_seq_length "${max_len}" \
        --num_train_epochs 0 \
        --no_metric

# ---

input_qa_file="${PREPROCESS_DATASET_DIR}/eval_predictions.json"

[[ RUN_QA_POSTPROCESS -eq '1' ]] && \
    echo '[*] Executing postprocessing QA output' && \
    python3.9 process_qa_output.py \
        --input_qa_file "${input_qa_file}" \
        --output_csv_file "${OUTPUT_CSV_PATH}"