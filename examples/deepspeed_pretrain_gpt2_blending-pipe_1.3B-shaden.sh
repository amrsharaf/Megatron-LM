#! /bin/bash

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

# Total GPUs
GPUS=256

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
# Make changes to the template JSON if you need to. We use sed to fill in the template
template_json="$script_dir/ds_config_gpt2_TEMPLATE.json"
config_json="$script_dir/ds_config_gpt2.json"


# CKPT is not currently used, but LOG_BASE should be set.
CKPT_BASE=/turing-ssd/users/${USER}/checkpoints/nvidia-diverge-debug/1.3B

# DeepSpeed tensorboarding; You still need to edit output_dir etc. in the TEMPLATE json.
TENSORBOARD_DIR="/data/users/shsmit/tensorboard/nvidia-diverge-debug"
TENSORBOARD_ENABLED="true"

# Appended to log names, etc.
EXP_NAME="1.3B-debug-fp32softmax-mb1"

gpt_options=""

# Data paths
BLEND="true"
DATA_PATH=/turing-ssd/users/bnorick/data/very_large_lm/indexed_datasets
CCN=$DATA_PATH/ccnet
BCR=$DATA_PATH/1_text_document
OWT=$DATA_PATH/2_text_document
STO=$DATA_PATH/3_text_document
RNS=$DATA_PATH/4_text_document
WIK=$DATA_PATH/5_text_document
CASE3="0.7900 ${CCN} 0.0046 ${BCR} 0.0759 ${OWT} 0.0267 ${STO} 0.0720 ${RNS} 0.0308 ${WIK}"
if [ "${BLEND}" = "true" ]; then
                #--vocab-file /data/users/bnorick/projects/very_large_lm/gpt2-vocab.json \
                #--merge-file /data/users/bnorick/projects/very_large_lm/gpt2-merges.txt \
        gpt_options="${gpt_options} \
                --vocab-file /data/users/bnorick/projects/turingNLG_100B/gpt2-vocab.json \
                --merge-file /data/users/bnorick/projects/turingNLG_100B/gpt2-merges.txt \
                --data-blend ${CASE3} \
                --data-impl mmap"
else
        gpt_options="${gpt_options} \
                --data-path /lustre/fsw/adlr-nlp/data/gpt2_indexed_dataset/roberta_dataset/rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_text_document \
		--vocab-file /lustre/fsw/adlr-nlp/data/gpt2_indexed_dataset/bpe/gpt2-vocab.json \
		--merge-file /lustre/fsw/adlr-nlp/data/gpt2_indexed_dataset/bpe/gpt2-merges.txt"
fi


#
# Parallelism parameters
#
PP=4 # 0 to disable
MP=2
DP=$(( ${GPUS} / ${MP} )) # default without PP
if [ ${PP} -gt 0 ]
then
        DP=$(( ${GPUS} / (${PP} * ${MP}) ))
fi


#
# Model parameters
#

HIDDEN=2048
HEADS=16
LAYERS=24


#
# Training parameters
#
SEQ_LEN=2048

# Batch size configuration

# Microbatch size per GPU
MBSIZE=1
TRAIN_BATCH_SIZE=512
GAS=$(( ${TRAIN_BATCH_SIZE} / (${MBSIZE} * ${DP}) ))

echo "batch=${TRAIN_BATCH_SIZE} microbatch=${MBSIZE} gas=${GAS}"

# Increase to 3.0e-4 when restarting with larger batch size
LR=2.0e-4


# Note that these are full model steps including $GAS gradient accumulation steps, so
# in total we'll train with samples=STEPS*TRAIN_BATCH_SIZE
STEPS=300000

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

CKPT_BASE=/turing-ssd/users/${USER}/checkpoints
#CHECKPOINT_PATH=${CKPT_BASE}/${EXP_NAME}/sq${SEQ_LEN}_pp${PP}_mp${MP}_dp${DP}_h${HEADS}_hd${HIDDEN}_ly${LAYERS}_bt${TRAIN_BATCH_SIZE}_lr${LR}
CHECKPOINT_PATH=/turing-ssd/users/shsmit/checkpoints/1.3B-debug-fp32softmax-mb1/sq2048_pp4_mp2_dp32_h16_hd2048_ly24_bt512_lr2.0e-4/

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
        --max-position-embeddings ${SEQ_LEN} \
        --batch-size ${MBSIZE} \
        --train-iters ${STEPS} \
        --seed ${SEED} \
        --lr-decay-iters 250000 \
        --eval-iters 40 \
        --log-interval 1 \
        --save-interval 4000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --eval-interval 2000 \
        --split 979,20,1 \
        --distributed-backend nccl \
        --lr ${LR} \
	--min-lr 2.0e-5 \
        --lr-decay-style cosine \
        --weight-decay 1.0e-1 \
        --clip-grad 1.0 \
        --warmup 0.0015 \
        --fp16 \
        --hysteresis 2 \
        --num-workers 2 \
        --apply-query-key-layer-scaling --bias-gelu-fusion --bias-dropout-fusion --scaled-upper-triang-masked-softmax-fusion \
        --override-lr-scheduler"
#        --attention_softmax_in_fp32



deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json}"
#deepspeed_options=""

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

if [ "${TENSORBOARD_ENABLED}" = "true" ]; then
    gpt_options="${gpt_options} --tensorboard-dir=${TENSORBOARD_DIR}/pp${PP}-mp${MP}-dp${DP}-gas${GAS}-zero${zero_stage}-${EXP_NAME}-restart4"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed ${WORKER_STR} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee -a $EXP_NAME.log
set +x

deepspeed ${WORKER_STR} ~/train.py