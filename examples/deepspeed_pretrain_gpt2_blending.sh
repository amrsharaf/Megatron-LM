#! /bin/bash

GPUS_PER_NODE=16
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=8
export DLWS_NUM_GPU_PER_WORKER=16

DATA_PATH=/turing-ssd/users/bnorick/data/very_large_lm/indexed_datasets
CHECKPOINT_PATH=/turing-nfs/users/arashari/checkpoints/sparse_sq2k_mp1_h16_hd2k_ly30_bt16_lr0.0003_lrdecay80k_beta0.95_wu0.01_zero2

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_2_config.json"

#Megatron Model Parallelism
mp_size=1

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=false
PA_CPU=false
CC=false
SYNCHRONIZE=false
PROFILE=false


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers 30 \
        --hidden-size 2048 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --batch-size 16 \
        --train-iters 100000 \
        --lr-decay-iters 80000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-blend 44 $DATA_PATH/ccnet 66 $DATA_PATH/megatron \
        --vocab-file /data/users/bnorick/projects/turingNLG_100B/gpt2-vocab.json \
        --merge-file /data/users/bnorick/projects/turingNLG_100B/gpt2-merges.txt \
        --data-impl mmap \
        --eval-iters 2000 \
        --log-interval 10 \
        --save-interval 2000 \
        --eval-interval 5000 \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --fp16 \
        --hysteresis 2 \
        --num-workers 0 \
        --apply_query_key_layer_scaling \
        --override-lr-scheduler \
"
#        --attention_softmax_in_fp32

       #Some Old Parameters
       #  --split 900,9,1 \
       #  --hidden-dropout 0.1 \
       #  --attention-dropout 0.1 \
       #  --hysteresis 2 \
       #  --num-workers 0 \
       #  --cache-dir /data/bert/users/corosset/ConversationalUnderstanding/Bert/checkpoints/MegatronGPT2/cache_bing \
       #   --save /turing-nfs/users/samyamr/checkpoints/tests \
       #  --train-data webtext \
       #  --resume-dataloader \
       #  --lazy-loader \
       #  --tokenizer-type GPT2BPETokenizer \
       #  --cache-dir cache \

 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs}
            "
#deepspeed_options=""

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="NCCL_TREE_THRESHOLD=0 deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

