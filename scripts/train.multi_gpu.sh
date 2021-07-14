#!/bin/bash
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4

export MASTER_PORT=6666
export MASTER_ADDR="localhost"

DEVICE_IDS=0,1,2,3

NUM_TRAIN_EPOCHS=10
PER_GPU_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1

CLASSIFIER_DROPOUT=0.0

PEAK_LR=5e-05
WARMUP_PROPORTION=0.6
MAX_GRAD_NORM=0.0
HF_MODEL_NAME="bert-base-cased"

BATCH_SIZE=`expr ${PER_GPU_TRAIN_BATCH_SIZE} \* ${GRADIENT_ACCUMULATION_STEPS} \* ${N_GPU_NODE}`

SAVE_CHECKPOINTS_DIR=checkpoints/${HF_MODEL_NAME}_E${NUM_TRAIN_EPOCHS}_B${BATCH_SIZE}_LR${PEAK_LR}_NORM${MAX_GRAD_NORM}/
SAVE_CHECKPOINTS_STEPS=1250

python -m torch.distributed.launch \
          --nproc_per_node ${N_GPU_NODE} \
          --nnodes ${N_NODES} \
          --node_rank ${NODE_RANK} \
          --master_addr ${MASTER_ADDR} \
          --master_port ${MASTER_PORT} \
          train.py --n_gpu ${WORLD_SIZE} \
                   --device_ids ${DEVICE_IDS} \
                   --save_checkpoints_dir ${SAVE_CHECKPOINTS_DIR} \
                   --save_checkpoints_steps ${SAVE_CHECKPOINTS_STEPS} \
                   --hf_model_name ${HF_MODEL_NAME} \
                   --num_train_epochs ${NUM_TRAIN_EPOCHS} \
                   --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
                   --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                   --learning_rate ${PEAK_LR} \
                   --warmup_proportion ${WARMUP_PROPORTION} \
                   --max_grad_norm ${MAX_GRAD_NORM} \
                   --classifier_dropout ${CLASSIFIER_DROPOUT};
