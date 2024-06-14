export WANDB_ENTITY=raquib-alam
export WANDB_PROJECT=llm-sft
LEARNING_RATE=0.000005
LOG_FILE=outputs/lr_${LEARNING_RATE}.log
CUDA_VISIBLE_DEVICES=2 nohup python train-v2.py \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --output_dir outputs/output_models_lr_${LEARNING_RATE} \
        --dataset "" \
        --dataset_format merlyn \
        --do_train True \
        --do_eval True \
        --do_mmlu_eval False \
        --source_max_len 6144 \
        --target_max_len 2048 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --logging_steps 1 \
        --num_train_epochs 5 \
        --save_strategy epoch \
        --data_seed 42 \
        --save_total_limit 10 \
        --evaluation_strategy epoch \
        --learning_rate $LEARNING_RATE \
        --report_to wandb \
        --optim paged_adamw_32bit > $LOG_FILE 2>&1 &