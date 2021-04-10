CHECKPOINT_DIR=checkpoints_dir/at-baseline-cutoff
python generate_distilled_data.py ../data-bin/iwslt14.tokenized.de-en-original \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --gen-subset train \
    --batch-size 250 --beam 5 --remove-bpe \
    --quiet \