CHECKPOINT_DIR=checkpoints_dir/ctc-intermediate_loss-cutoff-ffn_2048/
python average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --output $CHECKPOINT_DIR/checkpoint_best.pt \
    --num-update-checkpoints 5 \
