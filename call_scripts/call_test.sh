CHECKPOINT_DIR=checkpoints_dir/ctc-intermediate_loss-cutoff-ffn_2048/
python generate.py ../data-bin/iwslt14.tokenized.de-en-distilled-cutoff \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --quiet \
