
cs_model_path='./output_model/swag/pytorch_model.bin'
config_name='./output_model/swag/config.json'
tokenizer_name='./output_model/swag/'

train_file='./data/preprocess_dataset/swag_train.json'
valid_file='./data/preprocess_dataset/swag_valid.json'
test_file='./data/preprocess_dataset/swag_test.json'
test_cs_output='./data/preprocess_dataset/test_cs_output.json'

max_len=512
batch_size=1

echo '[*] Executing run_swag testing'
# python run_swag_no_trainer.py \
#     --model_name_or_path ${cs_model_path} \
#     --config_name ${config_name} \
#     --tokenizer_name ${tokenizer_name} \
#     --train_file ${train_file} \
#     --validation_file ${valid_file} \
#     --test_file ${test_file} \
#     --test_output_file ${test_cs_output} \
#     --per_device_eval_batch_size ${batch_size} \
#     --max_length ${max_len} \
#     --num_train_epochs 0

# ----------

dataset_dir='./data'
test_squad_file='./data/preprocess_dataset/squad_test.json'

echo '[*] Executing postprocessing CS output'
# python process_cs_output.py \
#     --input_data_dir ${dataset_dir} \
#     --input_cs_file ${test_cs_output} \
#     --output_squad_file ${test_squad_file}

# ----------

cs_model_path='./output_model/qa/pytorch_model.bin'
config_name='./output_model/qa/config.json'
tokenizer_name='./output_model/qa/'

train_file='./data/preprocess_dataset/squad_train.json'
valid_file='./data/preprocess_dataset/squad_valid.json'
test_file='./data/preprocess_dataset/squad_test.json'
test_squad_output_dir='./data/preprocess_dataset'

batch_size=1

echo '[*] Executing run_qa testing'
python run_qa_no_trainer.py \
    --model_name_or_path ${cs_model_path} \
    --config_name ${config_name} \
    --tokenizer_name ${tokenizer_name} \
    --train_file ${train_file} \
    --validation_file ${valid_file} \
    --do_predict \
    --test_file ${test_file} \
    --test_output_dir ${test_squad_output_dir} \
    --per_device_eval_batch_size ${batch_size} \
    --max_seq_length ${max_len} \
    --num_train_epochs 0