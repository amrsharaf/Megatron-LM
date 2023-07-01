#! /bin/bash

# Total GPUs
GPUS=512

# If we're running on a subset of the available resources, specify that here.
# https://www.deepspeed.ai/getting-started/#launching-deepspeed-training
WORKER_STR=""
#WORKER_STR="-e bad_node_to_exclude"

# For example, a single node execution would be:
#GPUS=8
#WORKER_STR="-i worker-0"


#
# Path configurations
#

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

# Make changes to the template JSON if you need to. We use sed to fill in the template
template_json="$script_dir/ds_config_gpt2_TEMPLATE.json"
config_json="$script_dir/ds_config_gpt2.json"


# CKPT is not currently used, but LOG_BASE should be set.
LOG_BASE=/turing-nfs/users/${USER}/logs
CKPT_BASE=/turing-nfs/users/${USER}/checkpoints


# DeepSpeed tensorboarding; You still need to edit output_dir etc. in the TEMPLATE json.
TENSORBOARD_ENABLED="false"

# Appended to log names, etc.
EXP_NAME="debug"

gpt_options=""

# Data paths
BLEND="false"
if [ "${BLEND}" = "true" ]; then
        DATA_PATH=/turing-ssd/users/bnorick/data/very_large_lm/indexed_datasets
        gpt_options="${gpt_options} \
                --vocab-file /data/users/bnorick/projects/turingNLG_100B/gpt2-vocab.json \
                --merge-file /data/users/bnorick/projects/turingNLG_100B/gpt2-merges.txt \
                --data-blend 44 $DATA_PATH/ccnet 66 $DATA_PATH/megatron \
                --data-impl mmap"
else
        DATA_PATH=./data/webtext/webtext_text_document
        gpt_options="${gpt_options} \
                --data-path $DATA_PATH \
                --vocab-file data/gpt2-vocab.json \
                --merge-file data/gpt2-merges.txt"
fi


#
# Parallelism parameters
#
PP=32 # 0 to disable
MP=8
DP=$(( ${GPUS} / ${MP} )) # default without PP
if [ ${PP} -gt 0 ]
then
        DP=$(( ${GPUS} / (${PP} * ${MP}) ))
fi


#
# Model parameters
#

HIDDEN=16384
HEADS=128

# Transformer layers per pipeline stage.
LAYERS_PER_STAGE=4

# Give the first/last stage one fewer layer for better load and memory balance
LAYERS=$(( (${LAYERS_PER_STAGE} * ${PP}) - 2 )) 

# Or we can use an equal number of layers per stage. This is sometimes imbalanced
# due to the embedding and loss computations on the first/last stage.
#LAYERS=$(( ${LAYERS_PER_STAGE} * ${PP} ))

# Or you can manually specify the number of layers
#LAYERS=104


#
# Training parameters
#
SEQ_LEN=2048

# Batch size configuration

# Microbatch size per GPU
MBSIZE=1

# If targeting pipeline efficiency for a given PP, we can do this:
# Selected 8 here for pipeline efficiency
GAS=$(( ${PP} * 8 ))
TRAIN_BATCH_SIZE=$(( ${GAS} * ${MBSIZE} * ${DP} ))

# Alternatively we can fix the effective batch size and work backwards
#TRAIN_BATCH_SIZE=2048
#GAS=$(( ${TRAIN_BATCH_SIZE} / (${MBSIZE} * ${DP}) ))

# Increase to 3.0e-4 when restarting with larger batch size
LR=1.5e-4


# Note that these are full model steps including $GAS gradient accumulation steps, so
# in total we'll train with samples=STEPS*TRAIN_BATCH_SIZE
#STEPS=10000
STEPS=100
#STEPS=20

SEED=1234

#
# ZeRO Configs - ZeRO is currently (temporarily) disabled and you should not touch these.
#
zero_stage=0
reduce_scatter=true
contigious_gradients=false
rbs=50000000
agbs=5000000000

# Actication Checkpointing and Contigious Memory
# Also don't touch these
chkp_layers=1
PA=true
PA_CPU=false # don't touch this
CC=false     # don't touch this
SYNCHRONIZE=false
PROFILE=false

CHECKPOINT_PATH=${CKPT_BASE}/sq${SEQ_LEN}_pp${PP}_mp${MP}_dp${DP}_h${HEADS}_hd${HIDDEN}_ly${LAYERS}_bt${TRAIN_BATCH_SIZE}_lr${LR}_lrdecay80k_beta0.95_wu0.01_zero${zero_stage}
LOGDIR=${LOG_BASE}/nvidia-prep/seq${SEQ_LEN}
mkdir -p ${LOGDIR}

LOGFILE="log-gpus${GPUS}_pp${PP}_mp${MP}_dp${DP}_h${HEADS}_hd${HIDDEN}_ly${LAYERS}_bt${TRAIN_BATCH_SIZE}_gas${GAS}_lr${LR}-${EXP_NAME}.txt"

sed "s/CONFIG_BATCH_SIZE/${TRAIN_BATCH_SIZE}/" ${template_json} \
        | sed "s/CONFIG_MBSIZE/${MBSIZE}/" \
        | sed "s/CONFIG_ZERO_STAGE/${zero_stage}/" \
        | sed "s/CONFIG_DP/${DP}/" \
        | sed "s/CONFIG_MP/${MP}/" \
        | sed "s/CONFIG_PP/${PP}/" \
        | sed "s/CONFIG_GAS/${GAS}/" \
        | sed "s/CONFIG_SEED/${SEED}/" \
        | sed "s/CONFIG_LR/${LR}/" \
        | sed "s/CONFIG_STEPS/${STEPS}/" \
        | sed "s/CONFIG_LAYERS/${LAYERS}/" \
        | sed "s/CONFIG_HIDDEN/${HIDDEN}/" \
        | sed "s/CONFIG_HEADS/${HEADS}/" \
        | sed "s/CONFIG_SEQ/${SEQ_LEN}/" \
        | sed "s/CONFIG_EXP_NAME/${EXP_NAME}/" \
        | sed "s/CONFIG_TB_ENABLED/${TENSORBOARD_ENABLED}/" \
        > ${config_json}


        # I have not yet tested the updated checkpointing codepaths in Megatron, so
        # temporarily disabled. In general checkpointing works though :-).
        #--save $CHECKPOINT_PATH \
        #--load $CHECKPOINT_PATH \
gpt_options=" \
        ${gpt_options} \
        --model-parallel-size ${MP} \
        --pipeline-parallel-size ${PP} \
        --gas ${GAS} \
        --num-layers ${LAYERS} \
        --hidden-size ${HIDDEN} \
        --num-attention-heads ${HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 2048 \
        --batch-size ${MBSIZE} \
        --train-iters ${STEPS} \
        --seed ${SEED} \
        --lr-decay-iters 80000 \
        --eval-iters 2000 \
        --log-interval 1 \
        --save-interval 2000 \
        --eval-interval 5000 \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr ${LR} \
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



deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${zero_stage} \
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

run_cmd="NCCL_TREE_THRESHOLD=0 deepspeed ${WORKER_STR} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee ${LOGFILE}

set +x
