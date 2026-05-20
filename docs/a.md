# 常用命令

## 数据集

ikea 数据集位置：`/data/ssd2/thw/data/dataset/ikea/`

## train

```sh
#!/bin/bash
# train_ikea_live1+.sh
export CUDA_VISIBLE_DEVICES=1,2
nohup \
torchrun --nproc_per_node=2 --standalone train.py \
    --deepspeed configs/deepspeed/zero2.json \
    --live_version live1+ \
    --train_datasets ikea_segment_qa_train ikea_narration_train \
    --eval_datasets ikea_segment_qa_test ikea_narration_test \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy epoch \
    --save_steps 1 \
    --learning_rate 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 2 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ikea/live1+ &> train.log &

```