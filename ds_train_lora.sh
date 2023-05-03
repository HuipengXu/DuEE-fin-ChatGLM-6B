DATA_PATH="DuEE_fin"

deepspeed lora.py \
    --deepspeed ds_config.json \
    --train_file $DATA_PATH/train.json \
    --validation_file $DATA_PATH/dev.json \
    --prompt_text_path $DATA_PATH/prompt.txt \
    --model_name_or_path chatglm-6b \
    --output_dir "output_dir" \
    --overwrite_output_dir \
    --fp16 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_source_length 1024 \
    --max_target_length 910 \
    --num_train_epochs 1 \
    --evaluation_strategy "steps" \
    --logging_steps 20 \
    --save_steps 20 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --bias "none"