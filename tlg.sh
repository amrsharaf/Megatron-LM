#!/bin/bash

set -euo pipefail

NUM_GPUS=16
NUM_NODES=2
SAVE_INTERVAL=500  # save every SAVE_INTERVAL iterations

DIR=$(pwd)
export TZ='America/Los_Angeles'
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
mkdir -p "$DIR"/logs

MP_SIZE=1 # model-parallel size
PP_SIZE=0 # pipeline-parallel size

PROXY_MODEL_NAME=mup.proxy-size_1.3b.hdim_256.nheads_2.seq-len_8192.pos_rope.yaml
BASE_SHAPES_PATH=${DIR}/assets/mup-assets/$PROXY_MODEL_NAME

# ===================
# Tokens Configuration
# ===================

B=1000000000 # 1B
TOKENS=$(( 30 * B ))  # 30B tokens

TARGET_GLOBAL_BATCH_SIZE=512 # global batch size
SEQ_LEN=8192

# ===================
# Gridsearch Configuration
# ===================
LR=0.01
LR_DECAY_STYLE=linear
MIN_LR=0.0
INIT_METHOD_STD=0.12
ATTN_TEMPERATURE=1.0
EMBEDDING_MULTIPLIER=10
OUTPUT_MULTIPLIER=1.0
LR_DECAY_RATIO_PERCENT=100

LOG_INTERVAL=100
EVAL_ITERS=10
EVAL_INTERVAL=500

# ===================
# LR Schedules Configurations
# ===================
TRAIN_ITERS_FOR_INDEXED_DATASET=$(( TOKENS / (TARGET_GLOBAL_BATCH_SIZE * SEQ_LEN) ))
TRAIN_ITERS=$(( TOKENS / (TARGET_GLOBAL_BATCH_SIZE * SEQ_LEN) ))

LR_DECAY_SAMPLES=$(( TOKENS * LR_DECAY_RATIO_PERCENT / (100 * SEQ_LEN) ))
LR_DECAY_ITERS=$(( LR_DECAY_SAMPLES / TARGET_GLOBAL_BATCH_SIZE ))  # This is an over-estimate, but should be okay (?)

WARMUP_SAMPLES=183105  # This is fixed based on GPT3 recommendations

SAVE_NORM_INTERVAL=5000000
PRECISION="bf16"
EXIT_DURATION=144000  # exit program after 14400 minutes = 10 days

if [ "${PRECISION}" == "fp16" ]; then
    FP16_ENABLED="true"
    BF16_ENABLED="false"
elif [ "${PRECISION}" == "bf16" ]; then
    FP16_ENABLED="false"
    BF16_ENABLED="true"
else
    echo "Invalid precision: ${PRECISION}, must be fp16 or bf16"
    exit 1
fi

# ===================
# Data configuration
# ===================

NFS_DATA_HOME="/turingnorwayeastpremium_data/datasets/tnlgv4/binarized/cl100k_base/"
TURING_PREMIUM_DATA_HOME="/turingnorwayeastpremium_data/datasets"

ARX="${NFS_DATA_HOME}/ArXiv_ftfy_cleaned_id_text_Document"
BOOKS="${NFS_DATA_HOME}/books.deduped.filtered_v0.1_Text_Document"
CC2020="${NFS_DATA_HOME}/CC-2020-50_id_cleaned_text_Document"
CC2021="${NFS_DATA_HOME}/CC-2021-04_id_cleaned_text_Document"
PCC="${NFS_DATA_HOME}/Pile-CC_id_cleaned_text_Document"
CC_RAI_FILTERED="${NFS_DATA_HOME}/cc-en.rai_filtered_v0.1.subsampled_0.3_Text_Document"

OWT2="${NFS_DATA_HOME}/OpenWebText2_ftfy_cleaned_id_text_Document"
GIT="${NFS_DATA_HOME}/Github_ftfy_id_text_Document"
NIH="${NFS_DATA_HOME}/NIH_ExPorter_ftfy_id_text_Document"
PM="${NFS_DATA_HOME}/PubMed_Abstracts_ftfy_id_text_Document"
SE="${NFS_DATA_HOME}/StackExchange_ftfy_id_text_Document"
WIK="${NFS_DATA_HOME}/Wikipedia_en_ftfy_id_text_Document"
RN="${NFS_DATA_HOME}/rn_dedup_shuf_cleaned_0.7_cleaned_text_Document"
RW="${TURING_PREMIUM_DATA_HOME}/refinedweb/rw_text_Document"

DATA_BLEND="0.01229 ${ARX} \
0.30785 ${BOOKS} \
0.38239 ${RW}  0.03848 ${PCC} \
0.04573 ${OWT2} \
0.01012 ${GIT} 0.00207 ${NIH} 0.02913 ${PM} 0.04215 ${SE} \
0.04669 ${WIK} 0.08676 ${RN}"

# ===================
# Model configuration
# ===================

BATCH_SIZE=2 # microbatch size

echo ${NUM_GPUS}
DP_SIZE=$(( NUM_GPUS / ((PP_SIZE > 0 ? PP_SIZE : 1) * MP_SIZE) )) # data-parallel size
GRAD_ACC_STEPS=$(( TARGET_GLOBAL_BATCH_SIZE / (BATCH_SIZE * DP_SIZE) ))

NAME="tlgv5_1.3b_30B-replace-cc-with-refined_web.gpus-$NUM_GPUS.seq_len${SEQ_LEN}.gbs${TARGET_GLOBAL_BATCH_SIZE}.mbs${BATCH_SIZE}"

CHECKPOINT_SAVE_TO="/turingnorwayeastpremium_data/users/amrsharaf/checkpoints/${NAME}"
TENSORBOARD_DIR="/scratch/turing_norwayeast_nfs/tensorboards/tlg/${NAME}"

# ===================
# Megatron parameters
# ===================

# NOTE (alonbenhaim): add --load argument below ("--load ${CHECKPOINT_LOAD_FROM} \") if you want to continue pretraining from checkpoint
megatron_options=" \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--model-parallel-size ${MP_SIZE} \
--init-method-std ${INIT_METHOD_STD} \
--lr-decay-samples ${LR_DECAY_SAMPLES} \
--warmup-samples ${WARMUP_SAMPLES} \
--batch-size ${BATCH_SIZE} \
--use-checkpoint-lr-scheduler \
--exit-duration ${EXIT_DURATION} \
--target-global-batch-size ${TARGET_GLOBAL_BATCH_SIZE} \
--num-layers 24 \
--hidden-size 2048 \
--num-attention-heads 16 \
--seq-length ${SEQ_LEN} \
--max-position-embeddings ${SEQ_LEN} \
--train-iters ${TRAIN_ITERS} \
--train-iters-for-indexed-dataset ${TRAIN_ITERS_FOR_INDEXED_DATASET} \
--lr-decay-iters ${LR_DECAY_ITERS} \
--lr ${LR} \
--positional-embedding-type rope \
--min-lr ${MIN_LR} \
--lr-decay-style ${LR_DECAY_STYLE} \
--tokenizer-type CL100kBaseBPETokenizer \
--data-blend ${DATA_BLEND} \
--blend-mode heterogeneous \
--split 98,2,0 \
--log-interval ${LOG_INTERVAL} \
--save-norm-interval ${SAVE_NORM_INTERVAL} \
--eval-interval ${EVAL_INTERVAL} \
--eval-iters ${EVAL_ITERS} \
--save-interval ${SAVE_INTERVAL} \
--save ${CHECKPOINT_SAVE_TO} \
--weight-decay 0.1 \
--clip-grad 1.0 \
--tensorboard-dir ${TENSORBOARD_DIR} \
--hysteresis 2 \
--num-workers 0 \
--${PRECISION} \
--bias-gelu-fusion \
--use-flash-attn \
--checkpoint-activations \
--bias-dropout-fusion \
--scaled-upper-triang-masked-softmax \
--kernels-build-path /tmp/megatron-kernels \
--build-kernels-per-node \
--mup-use-parametrization \
--mup-base-shapes-path ${BASE_SHAPES_PATH} \
--mup-use-scaling \
--mup-output-multiplier ${OUTPUT_MULTIPLIER} \
--mup-embedding-multiplier ${EMBEDDING_MULTIPLIER} \
--mup-attn-multiplier ${ATTN_TEMPERATURE}
"

# ====================
# DeepSpeed parameters
# ====================

ZERO_STAGE=1
PRESCALE_GRADIENTS=false

template_json="${DIR}/examples/ds_config_gpt2_TEMPLATE.json"
config_json="${DIR}/examples/ds_config_gpt2_megatron_deepspeed_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${TARGET_GLOBAL_BATCH_SIZE}/" "${template_json}" \
| sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
| sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
| sed "s/CONFIG_FP16_ENABLED/${FP16_ENABLED}/" \
| sed "s/CONFIG_BF16_ENABLED/${BF16_ENABLED}/" \
| sed "s/\"stage\": 0/\"stage\": ${ZERO_STAGE}/" \
| sed "s/\"prescale_gradients\": true/\"prescale_gradients\": ${PRESCALE_GRADIENTS}/" \
> "${config_json}"

deepspeed_options=" \
--deepspeed \
--deepspeed_config ${config_json} \
--gas ${GRAD_ACC_STEPS} \
--pipeline-parallel-size ${PP_SIZE} \
--partition-activations"

# Print the command before running
echo "deepspeed --num_gpus $(( NUM_GPUS / NUM_NODES )) --num_nodes ${NUM_NODES} \"${DIR}\"/pretrain_gpt.py ${megatron_options} ${deepspeed_options}"
deepspeed --num_gpus $(( NUM_GPUS / NUM_NODES )) --num_nodes ${NUM_NODES} "${DIR}"/pretrain_gpt.py ${megatron_options} ${deepspeed_options}
