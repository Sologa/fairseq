CHECKPOINT_DIR=checkpoints_dir/at-baseline-cutoff
python generate.py ../data-bin/iwslt14.tokenized.de-en-original \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --quiet \