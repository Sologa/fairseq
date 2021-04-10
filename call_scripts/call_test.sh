CHECKPOINT_DIR=checkpoints_dir/ctc-intermediate_loss/
python generate.py ../data-bin/iwslt14.tokenized.de-en-distilled \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --quiet \