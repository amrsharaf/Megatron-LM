#! /bin/bash
# set -euo pipefail

SEQ_LENGTH=2048

# ==================
# 530B Model on A100
# ==================
# MP=8
# PP=4
# LAYERS=105
# HIDDEN_SIZE=20480
# NUM_ATTN_HEADS=128
# CHECKPOINT_PATH=/vc_data_blob/users/zhunliu/checkpoints/FromNvidia/gpt3-530b-lr5em5
# ITERATION=0080163


# =======================
# 530B Model on V100-32GB
# =======================
# MP=8
# PP=6
# LAYERS=105
# HIDDEN_SIZE=20480
# NUM_ATTN_HEADS=128
# CHECKPOINT_PATH=/vc_data_blob/users/zhunliu/checkpoints/FromNvidia/gpt3-530b-lr5em5
# ITERATION=0080163

# ==================
# 6.7B Model on A100
# ==================
# MP=4
# PP=2
# LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# CHECKPOINT_PATH=/vc_data_blob/users/adatkins/checkpoints/FromNvidia/gpt3-6.7b-bf16-megatron-nd-other-21-pipe/
# ITERATION=0152085

# ==================
# 1.3B Model on A100
# ==================
# MP=2
# PP=1
# LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# CHECKPOINT_PATH=/vc_data_blob/users/zhunliu/checkpoints/FromNvidia/gpt3-1.3b-bf16-deepspeed-nd-other-21-mp2pp4-w0
# ITERATION=0058329

# ==================
# 204M Model on A100
# ==================
MP=2
PP=2
LAYERS=12
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
# CHECKPOINT_PATH=/vc_data_blob/users/xihlin/tnlgv2/pretrain/tnlgv2-204m-mp2pp2-64gpus-pretrain-lr6-flash-ds0.7.4-resizable-ckpt
CHECKPOINT_PATH=/vc_data_blob/users/xihlin/tnlgv2/pretrain/tnlgv2-204m-mp2pp2-64gpus-pretrain-lr6-flash-ds0.7.4-resizable-ckpt
ITERATION=0500000
SEQ_LENGTH=1024

# ================

# add TNLGv2 code repo to PYTHONPATH
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TNLG_DIR=`realpath $SCRIPT_DIR/..`
export PYTHONPATH=$PYTHONPATH:$TNLG_DIR


# install netbase if it's not present
if ! (apt list --installed | grep netbase) > /dev/null 2>&1; then
    ds_ssh "sudo apt-get update"
    ds_ssh "sudo apt-get -o Dpkg::Options::=\"--force-confmiss\" install --reinstall netbase"
fi

# launch model & demo in the background and write logs to a file
mkdir -p logs
curr_datetime=$(date +%F_%H-%M-%S)
model_log_name=model_backend.${curr_datetime}.log
demo_log_name=demo_frontend.${curr_datetime}.log
touch logs/"$model_log_name"
touch logs/"$demo_log_name"


GPUS_NEEDED=$(( MP * PP ))

if [[ ${GPUS_NEEDED} < ${DLWS_NUM_GPU_PER_WORKER} ]]; then
    NUM_GPUS=${GPUS_NEEDED}
    NUM_NODES=1
else
    NUM_GPUS=${DLWS_NUM_GPU_PER_WORKER}
    NUM_NODES=$(( GPUS_NEEDED / DLTS_NUM_GPU_PER_WORKER ))
fi

# Soft-linking the checkpoint to a local directory
# so that we can write our own latest_checkpointed_iteration.txt without interfering with the original one
TEMP_CHECKPOINT_PATH=/tmp/tnlgv2_checkpoint_link/
ds_ssh "mkdir -p ${TEMP_CHECKPOINT_PATH}"
ds_ssh "rm -rf ${TEMP_CHECKPOINT_PATH}/*"
ds_ssh "ln -s ${CHECKPOINT_PATH}/iter_${ITERATION} ${TEMP_CHECKPOINT_PATH}/iter_${ITERATION}"
ds_ssh "echo ${ITERATION} > ${TEMP_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"

NCCL_ASYNC_ERROR_HANDLING=0 \
PYTHONIOENCODING=utf8 \
deepspeed --num_gpus="${NUM_GPUS}" --num_nodes="${NUM_NODES}" tools/generate_samples_gpt2.py \
       --model-parallel-size "${MP}" \
       --pipeline-parallel-size "${PP}" \
       --deepspeed \
       --deepspeed_config assets/ds_config_gen_fp16.json \
       --num-layers "${LAYERS}" \
       --hidden-size "${HIDDEN_SIZE}" \
       --num-attention-heads "${NUM_ATTN_HEADS}" \
       --tokenizer-type GPT2BPETokenizer \
       --batch-size 1 \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --out-seq-length 256 \
       --temperature 1.0 \
       --vocab-file ./assets/gpt2_vocab.json \
       --merge-file ./assets/gpt2_merges.txt \
       --load "${TEMP_CHECKPOINT_PATH}" \
       --num-samples 0 \
       --fp16 \
       --top_p 1.0 \
       --sample-input-file assets/example_inputs.txt \
       --sample-output-file example_outputs.txt \
       --no-load-rng
       

# some info
echo "API in fp16 mode launched. Model backend logs are written to $(pwd)/logs/model_backend.latest.log, demo frontend logs are written to $(pwd)/logs/demo_frontend.latest.log. You can do \`tail -F <logfile>\` to monitor the logs."
