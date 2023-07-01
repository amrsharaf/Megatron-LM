#!/bin/bash
# set -euo pipefail

# This script assumes these environment variables are set:
# MP - number of model parallelism
# PP - number of parameter parallelism
# LAYERS - number of layers
# HIDDEN_SIZE - hidden size
# CHECKPOINT_PATH - path to checkpoint
# NUM_ATTN_HEADS - number of attention heads
# ITERATION - iteration number
# DLTS_NUM_GPU_PER_WORKER - number of GPUs per worker

GPUS_NEEDED=$(( MP * PP ))

if [[ ${GPUS_NEEDED} < ${DLWS_NUM_GPU_PER_WORKER} ]]; then
    NUM_GPUS=${GPUS_NEEDED}
    NUM_NODES=1
else
    NUM_GPUS=${DLWS_NUM_GPU_PER_WORKER}
    NUM_NODES=$(( GPUS_NEEDED / DLTS_NUM_GPU_PER_WORKER ))
fi

export PYTHONPATH=$PYTHONPATH:/work/workspace/TNLGv2:/work/tmp/libs

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
       --out-seq-length 1536 \
       --temperature 1.0 \
       --vocab-file ./assets/gpt2_vocab.json \
       --merge-file ./assets/gpt2_merges.txt \
       --load "${TEMP_CHECKPOINT_PATH}" \
       --num-samples 0 \
       --fp16 \
       --model-instance \
       --uptime-port 13382 \
       --demo-model turingNLG.freeform \
       --top_p 1.0 \
       --no-load-rng
