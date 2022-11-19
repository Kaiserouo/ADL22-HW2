
# whether to run those operation, set to 1 if want to run
DO_PREPROCESS_DATASET=1
DO_TRAIN_SWAG=1
DO_TRAIN_QA=1

# data directory, containing {context, train, valid}.json
DATA_DIR='./data'
# directory for storing preprocessed data
PREPROCESS_DATASET_DIR='./output_model/preprocess_dataset'
# CS and QA model directory
SWAG_DIR='./output_model/swag'
QA_DIR='./output_model/qa'

# ---------

echo '[*] Generating Directories'
mkdir -p "${PREPROCESS_DATASET_DIR}"
mkdir -p "${SWAG_DIR}"
mkdir -p "${QA_DIR}"

[[ DO_PREPROCESS_DATASET -eq '1' ]] && \
    echo '[*] Executing preprocess_dataset' && \
    python3.9 preprocess_dataset.py \
        --input_data_dir "${DATA_DIR}" \
        --output_data_dir "${PREPROCESS_DATASET_DIR}"

# ---------

# paths for loading models. you have to put in the arguments yourself
cs_model_path="${SWAG_DIR}/pytorch_model.bin"
config_name="${SWAG_DIR}/config.json"
tokenizer_name="${SWAG_DIR}"

model="hfl/chinese-roberta-wwm-ext"
train_file="${PREPROCESS_DATASET_DIR}/swag_train.json"
valid_file="${PREPROCESS_DATASET_DIR}/swag_valid.json"
output_dir="${SWAG_DIR}"

# training setting
max_len=512

# hyperparameter setting
lr=1e-5
weight_decay=0
num_epoch=1
batch_size=1
grad_acc_step=2   # effective batch size = batch_size * grad_acc_step

[[ DO_TRAIN_SWAG -eq '1' ]] && \
    echo '[*] Executing run_swag' && \
    python3.9 run_swag_no_trainer.py \
        --model_name_or_path "${model}" \
        --train_file "${train_file}" \
        --validation_file "${valid_file}" \
        --max_length "${max_len}" \
        --per_device_train_batch_size "${batch_size}" \
        --per_device_eval_batch_size "${batch_size}" \
        --learning_rate "${lr}" \
        --weight_decay "${weight_decay}" \
        --num_train_epochs "${num_epoch}" \
        --gradient_accumulation_steps "${grad_acc_step}" \
        --output_dir "${output_dir}" \
        --with_tracking

# --------------

# paths for loading models. you have to put in the arguments yourself
qa_model_path="${QA_DIR}/pytorch_model.bin"
config_name="${QA_DIR}/config.json"
tokenizer_name="${QA_DIR}"

model="hfl/chinese-roberta-wwm-ext"
train_file="${PREPROCESS_DATASET_DIR}/squad_train.json"
valid_file="${PREPROCESS_DATASET_DIR}/squad_valid.json"
output_dir="${QA_DIR}"

# training setting
max_len=512

# hyperparameter setting
lr=2e-5
weight_decay=0
num_epoch=3
batch_size=1
grad_acc_step=2   # effective batch size = batch_size * grad_acc_step

# how many steps per one record
record_steps=2000

[[ DO_TRAIN_QA -eq '1' ]] && \
    echo '[*] Executing run_qa_no_trainer' && \
    python3.9 run_qa_no_trainer.py \
        --model_name_or_path "${model}" \
        --train_file "${train_file}" \
        --validation_file "${valid_file}" \
        --max_seq_length "${max_len}" \
        --per_device_train_batch_size "${batch_size}" \
        --per_device_eval_batch_size "${batch_size}" \
        --learning_rate "${lr}" \
        --weight_decay "${weight_decay}" \
        --num_train_epochs "${num_epoch}" \
        --gradient_accumulation_steps "${grad_acc_step}" \
        --output_dir "${output_dir}" \
        --record_steps "${record_steps}" \
        --with_tracking
        