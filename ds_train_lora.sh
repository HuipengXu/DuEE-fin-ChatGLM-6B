DATA_PATH = "DuEE_fin"

deepspeed lora.py \
    --deepspeed ds_config.json \
    --train_file $DATA_PATH/train.json \
    --validation_file $DATA_PATH/dev.json \
    --model_name_or_path chatglm-6b \
    --output_dir "output_dir" \
    --overwrite_output_dir \
    --fp16 \
    --do_train \
    --do_eval \
    --max_source_length 1024 \
    --max_target_length 910 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --bias "none" \
    --task_type "CAUSAL_LM"

