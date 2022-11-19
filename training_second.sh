NUM=4

for i in $(seq 1 ${NUM}); do
    mkdir -p "./output_model/swag/${i}"

    model='hfl/chinese-roberta-wwm-ext'
    train_file='./data/preprocess_dataset/swag_train.json'
    valid_file='./data/preprocess_dataset/swag_valid.json'
    output_dir="./output_model/swag/${i}"
    
    echo "[*] Iteration ${i}: output dir ${output_dir}"

    # training setting
    max_len=512

    # hyperparameter setting
    lr=1e-5
    weight_decay=0
    num_epoch=1
    batch_size=1
    grad_acc_step=2   # effective batch size = batch_size * grad_acc_step

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
        
done
