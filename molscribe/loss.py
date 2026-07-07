import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .diffusion.edge_loss import EdgeDiffusionLoss
from .tokenizer import PAD_ID, MASK, MASK_ID


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        log_probs = F.log_softmax(output, dim=-1)

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # reduction mean or sum?
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        loss = self.criterion(output, target)
        return loss


class GraphLoss(nn.Module):

    def __init__(self, use_edge_diffusion_loss=False):
        super(GraphLoss, self).__init__()
        weight = torch.ones(7) * 10
        weight[0] = 1
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)
        self.edge_diffusion_criterion = EdgeDiffusionLoss() if use_edge_diffusion_loss else None

    def forward(self, outputs, targets):
        """
        outputs['edges']: [B, 7, K, K] baseline graph logits.
        targets['edges']: [B, K, K] baseline graph targets.

        Optional diffusion path, enabled only with use_edge_diffusion_loss:
        outputs['edge_diffusion']: [B, 7, K, K] denoising logits.
        targets['edge_diffusion']: [B, K, K] sparse denoising targets where
        ignored entries are -100.
        """
        results = {}
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        if 'edge_diffusion' in outputs:
            if self.edge_diffusion_criterion is None:
                raise ValueError("edge_diffusion outputs require use_edge_diffusion_loss")
            results['edge_diffusion'] = self.edge_diffusion_criterion(
                outputs['edge_diffusion'],
                targets['edge_diffusion'],
            )
        return results


class Criterion(nn.Module):

    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        for format_ in args.formats:
            if format_ == 'edges':
                criterion['edges'] = GraphLoss(use_edge_diffusion_loss=getattr(args, 'use_edge_diffusion_loss', False))
            else:
                if MASK in tokenizer[format_].stoi:
                    ignore_indices = [PAD_ID, MASK_ID]
                else:
                    ignore_indices = []
                criterion[format_] = SequenceLoss(args.label_smoothing, len(tokenizer[format_]),
                                                  ignore_index=PAD_ID, ignore_indices=ignore_indices)
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, results, refs):
        losses = {}
        for format_ in results:
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_](predictions, targets)
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                if loss_.numel() > 1:
                    loss_ = loss_.mean()
                losses[format_] = loss_
        return losses
