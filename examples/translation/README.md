# Neural Machine Translation

This README contains instructions for [using pretrained translation models](#example-usage-torchhub)
as well as [training new models](#training-a-new-model).


## Example usage (torch.hub)

We require a few additional Python dependencies for preprocessing:
```bash
pip install fastBPE sacremoses subword_nmt
```

## Example usage (CLI tools)

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```bash
mkdir -p data-bin
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
python generate.py data-bin/wmt14.en-fr.newstest2014  \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
# ...
# | Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
# | Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
# BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

## Training a new model

### IWSLT'14 German to English (Transformer)

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
python preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 8 --joined_dictionary \
```

Next we'll train a Transformer translation model over this data:
```bash
CHECKPOINT_DIR=AT_iwslt14_de_en_baseline
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-update 2000000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --eval-tokenized-bleu \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints_dir/$CHECKPOINT_DIR \
```

Averaging the best 5 checkpoints during validation (optional):
```bash
CHECKPOINT_DIR=checkpoints_dir/AT_iwslt14_de_en_baseline
python average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --output $CHECKPOINT_DIR/checkpoint_best.pt \
    --num-update-checkpoints 5 \
```

Finally we can evaluate our trained model:
```bash
CHECKPOINT_DIR=checkpoints_dir/AT_iwslt14_de_en_baseline
python generate.py data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```


### WMT'14 English to French
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_fr
python preprocess.py \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

# Train the model
mkdir -p checkpoints/fconv_wmt_en_fr
python train.py \
    data-bin/wmt14_en_fr \
    --arch fconv_wmt_en_fr \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 3000 \
    --save-dir checkpoints/fconv_wmt_en_fr

# Evaluate
python generate.py \
    data-bin/fconv_wmt_en_fr \
    --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

## Multilingual Translation

We also support training multilingual translation models. In this example we'll
train a multilingual `{de,fr}-en` translation model using the IWSLT'17 datasets.

Note that we use slightly different preprocessing here than for the IWSLT'14
En-De data above. In particular we learn a joint BPE code for all three
languages and use fairseq-interactive and sacrebleu for scoring the test set.

```bash
# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

# Then download and preprocess the data
cd examples/translation/
bash prepare-iwslt17-multilingual.sh
cd ../..

# Binarize the de-en dataset
TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
mkdir -p checkpoints/multilingual_transformer
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=legacy_ddp \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens 4000 \
    --update-freq 8

# Generate and score the test set with sacrebleu
SRC=de
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
      --task multilingual_translation --lang-pairs de-en,fr-en \
      --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
grep ^H iwslt17.test.${SRC}-en.en.sys | cut -f3 \
    | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en
```

##### Argument format during inference

During inference it is required to specify a single `--source-lang` and
`--target-lang`, which indicates the inference langauge direction.
`--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
the same value as training.
