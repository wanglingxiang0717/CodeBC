#!/bin/bash
port=$(shuf -i25000-30000 -n1)
output_dir=${}
if [ ! -d ${output_dir} ];then  
    mkdir ${output_dir}
fi
deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
    --data_path data_temp \
    --dataset_name ${} \
    --model_name_or_path ${} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 2048 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 42 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method lora \
    --offload \
    --output_dir ${output_dir} \
    | tee ${output_dir}/train.log