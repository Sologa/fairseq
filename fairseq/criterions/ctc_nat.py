# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length. Should be set for CTC NAT since we have no idea if this condition holds."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    cutoff: bool = field(
        default=False,
        metadata={"help": "Apply cutoff data augmentation."},
    )
    cutoff_regularization_loss: float = field(
        default=5.0,
        metadata={
            "help": "Cutoff regularization coefficient."
        },
    )

    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc_nat", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.task = task

        # for beam decoding later
        '''
        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None
        '''

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        self.cutoff = cfg.cutoff
        self.cutoff_regularization_loss = cfg.cutoff_regularization_loss

        self.cross_entropy = LabelSmoothedCrossEntropyCriterion(task, cfg.sentence_avg, 0.1)

    def forward(self, model, sample, reduce=True):
        ctc_out, primary_enc_out, encoder_padding_mask, dec_output = model(**sample["net_input"])

        lprobs = list()
        for i in range(len(ctc_out)):
            lprobs.append(model.get_normalized_probs(
                [ctc_out[i]], log_probs=True
            ).contiguous())  # (T, B, C) from the encoder

        input_lengths = sample["net_input"]["src_lengths"]
       

        upsample_coefficient = int(model.encoder.upsample_coefficient)
        input_lengths *= upsample_coefficient

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs[-1],
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            for i in range(len(lprobs)-1):
                loss += F.ctc_loss(
                    lprobs[i],
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )

        if self.cutoff:
            orig_tokens = sample["net_input"]['src_tokens'].clone()
            sample["net_input"]['src_tokens'] = self.task._mask_tokens(sample["net_input"]['src_tokens'])
            ctc_out, secondary_enc_out, encoder_padding_mask, dec_output = model(**sample["net_input"])

            lprobs = list()
            for i in range(len(ctc_out)):
                lprobs.append(model.get_normalized_probs(
                    [ctc_out[i]], log_probs=True
                ).contiguous())  # (T, B, C) from the encoder

            with torch.backends.cudnn.flags(enabled=False):
                loss += F.ctc_loss(
                    lprobs[-1],
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )
                for i in range(len(lprobs)-1):
                    loss += F.ctc_loss(
                        lprobs[i],
                        targets_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="sum",
                        zero_infinity=self.zero_infinity,
                    )

            loss += self.cutoff_regularization_loss * self.compute_regularization_loss(model, primary_enc_out, secondary_enc_out, reduce=reduce)

        if dec_output is not None:
            dec_loss, nll_loss = self.cross_entropy.compute_loss(model, dec_output, sample, reduce=reduce)
            loss += dec_loss

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_regularization_loss(self, model, primary_net_output, secondary_net_output, pad_mask=None, reduce=True):
        mean_net_output = (primary_net_output[0] + secondary_net_output[0]) / 2
        m = model.get_normalized_probs((mean_net_output,), log_probs=False)
        p = model.get_normalized_probs(primary_net_output, log_probs=True)
        q = model.get_normalized_probs(secondary_net_output, log_probs=True)

        primary_loss = F.kl_div(p, m, reduction='none')
        secondary_loss = F.kl_div(q, m, reduction='none')
        if pad_mask is not None:
            primary_loss.masked_fill_(pad_mask, 0.)
            secondary_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            primary_loss = primary_loss.sum()
            secondary_loss = secondary_loss.sum()

        loss = (primary_loss + secondary_loss) / 2
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
