# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_regularization')
class LabelSmoothedCrossEntropyCriterionWithRegularization(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, regularization_weight=5.0):
        super().__init__(task, sentence_avg, label_smoothing)
        self.regularization_weight = regularization_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--regularization_weight', default=1.0, type=float, metavar='D',
                            help='weight for the regularization loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        primary_net_output = model(**sample["net_input"])
        primary_loss, primary_nll_loss = self.compute_loss(model, primary_net_output, sample, reduce=reduce)

        orig_tokens = sample["net_input"]['src_tokens'].clone()
        orig_prev_tokens = sample['net_input']['prev_output_tokens'].clone()
        sample["net_input"]['src_tokens'] = self.task._mask_tokens(sample["net_input"]['src_tokens'])
        sample['net_input']['prev_output_tokens'] = self.task._mask_tokens(sample['net_input']['prev_output_tokens'])
        secondary_net_output = model(**sample["net_input"])
        secondary_loss, secondary_nll_loss = self.compute_loss(model, secondary_net_output, sample, reduce=reduce)

        sample["net_input"]['src_tokens'] = orig_tokens
        sample['net_input']['prev_output_tokens'] = orig_prev_tokens

        pad_mask = sample['target'].eq(self.padding_idx).unsqueeze(-1).repeat(1, 1, secondary_net_output[0].shape[-1]) 

        regularization_loss = self.compute_regularization_loss(model, primary_net_output, secondary_net_output, pad_mask=pad_mask, reduce=reduce)


        loss = primary_loss + secondary_loss + self.regularization_weight * regularization_loss
        nll_loss = primary_nll_loss + secondary_nll_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
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
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        regularization_loss_sum = utils.item(sum(log.get('regularization_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('regularization_loss', regularization_loss_sum / sample_size, sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
