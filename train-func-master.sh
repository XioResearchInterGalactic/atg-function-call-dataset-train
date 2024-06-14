LEARNING_RATE=0.000005
LOG_FILE=outputs/lr_${LEARNING_RATE}.log
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
        --model_name_or_path allyson-ai/FuncMaster-v0.1-Mistral-7B \
        --output_dir outputs/output_models_lr_${LEARNING_RATE} \
        --dataset "" \
        --dataset_format merlyn \
        --do_train True \
        --do_eval False \
        --do_mmlu_eval False \
        --source_max_len 4096 \
        --target_max_len 4096 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --logging_steps 1 \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --data_seed 42 \
        --save_total_limit 40 \
        --evaluation_strategy steps \
        --eval_dataset_size 0 \
        --max_eval_samples 10 \
        --eval_steps 10000000 \
        --learning_rate $LEARNING_RATE \
        --report_to wandb \
        --optim paged_adamw_32bit > $LOG_FILE 2>&1 &