DO_PREPROCESS_DATASET=0
DO_TRAIN_SWAG=1
# SWAG_LOAD_MODEL=1
DO_TRAIN_QA=1
# QA_LOAD_MODEL=1

echo '[*] Generating Directories'
mkdir -p ./data/preprocess_dataset
mkdir -p ./output_model/qa
mkdir -p ./output_model/swag
mkdir -p ./output_model/tmp_files


[[ DO_PREPROCESS_DATASET -eq '1' ]] && \
    echo '[*] Executing preprocess_dataset' && \
    python preprocess_dataset.py \
        --input_data_dir ./data \
        --output_data_dir ./data/preprocess_dataset

# ---------

# paths
cs_model_path='./output_model/swag/pytorch_model.bin'
config_name='./output_model/swag/config.json'
tokenizer_name='./output_model/swag/'

model='hfl/chinese-macbert-base'
train_file='./data/preprocess_dataset/swag_train.json'
valid_file='./data/preprocess_dataset/swag_valid.json'
output_dir='./output_model/swag'

# training setting
max_len=512

# hyperparameter setting
lr=3e-5
weight_decay=0
num_epoch=1
batch_size=1
grad_acc_step=2   # effective batch size = batch_size * grad_acc_step

[[ DO_TRAIN_SWAG -eq '1' ]] && \
    echo '[*] Executing run_swag' && \
    python run_swag_no_trainer.py \
        --model_name_or_path ${model} \
        --train_file ${train_file} \
        --validation_file ${valid_file} \
        --max_length ${max_len} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --weight_decay ${weight_decay} \
        --num_train_epochs ${num_epoch} \
        --gradient_accumulation_steps ${grad_acc_step} \
        --output_dir ${output_dir} \
        --with_tracking

# --------------

# paths
qa_model_path='./output_model/qa/pytorch_model.bin'
config_name='./output_model/qa/config.json'
tokenizer_name='./output_model/qa/'

model='hfl/chinese-macbert-base'
train_file='./data/preprocess_dataset/squad_train.json'
valid_file='./data/preprocess_dataset/squad_valid.json'
output_dir='./output_model/qa'

# training setting
max_len=512

# hyperparameter setting
lr=3e-5
weight_decay=0
num_epoch=1
batch_size=1
grad_acc_step=2   # effective batch size = batch_size * grad_acc_step

[[ DO_TRAIN_QA -eq '1' ]] && \
    echo '[*] Executing run_qa_no_trainer' && \
    python run_qa_no_trainer.py \
        --model_name_or_path ${model} \
        --train_file ${train_file} \
        --validation_file ${valid_file} \
        --max_seq_length ${max_len} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --weight_decay ${weight_decay} \
        --num_train_epochs ${num_epoch} \
        --gradient_accumulation_steps ${grad_acc_step} \
        --output_dir ${output_dir} \
        --record_steps 2000 \
        --with_tracking
        