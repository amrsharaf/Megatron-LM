#! /bin/bash

# Runs the "345M" parameter model with incremental batch size
# Compare with pretrain_gpt2.sh to see the difference

# To use incremental batch size the following changes need to be made
# in the script.

# Assume the target gloabl batch size is 512 and you want to run incremental
# global batch size of 256, then all the intervals (save, exit, eval and log)
# should be doubled (as batch size lowered by 2). Also the train-iters
# and lr-decay-iters need to be increased by a factor of two. Three
# additional parameters need to be added, batch-size-increase,
# batch-size-increase-iter (the number of iterations you want to run
# using batch size of 256, and what is the target global batch 
# size (512 in this case).

RANK=0
WORLD_SIZE=1

DATA_PATH=data/webtext/webtext_text_document
CHECKPOINT_PATH=/turing-nfs/users/samyamr/checkpoints/


python pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --batch-size-increase \
       --batch-size-increase-iter 1000 \
       --target-global-batch-size 512
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000000 \
       --lr-decay-iters 640000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 200 \
       --save-interval 20000 \
       --eval-interval 2000 \
       --eval-iters 10 \
       --fp16


set +x
