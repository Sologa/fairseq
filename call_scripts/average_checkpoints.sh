CHECKPOINT_DIR=checkpoints_dir/at-baseline-cutoff
python average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --output $CHECKPOINT_DIR/checkpoint_best.pt \
    --num-update-checkpoints 5 \