from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import os
import math

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')): 
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth')) 
    else: 
        val_pred, val_target = tester.calc_logit(val_loader) 
        test_pred, test_target = tester.calc_logit(test_loader) 
        torch.save((val_pred, val_target, test_pred, test_target), 
                    os.path.join(args.save, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    with open(os.path.join(args.save, 'dynamic.txt'), 'w') as fout:
#        for p in range(1, 40):
#            print("*********************")
#            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
#            probs = torch.exp(torch.log(_p) * torch.range(1, args.nBlocks))
#            probs /= probs.sum()
#            acc_val, _, T = tester.dynamic_eval_find_threshold(
#                val_pred, val_target, probs, flops)
#            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
#                test_pred, test_target, flops, T)
#            print('T: ', list(T.cpu().numpy()))
#            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops / 1e6))
#            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))

        
        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            # Same thresholding
            T = [threshold] * (len(flops) - 1) + [-100000000.0]
            #T = [0.79204506, 0.5818701, 0.4145896, 0.24300988, -100000000.0]
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                test_pred, test_target, flops, T)
            print('T: ', T)
            print('test acc: {:.3f}, test flops: {:.2f}M'.format(acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))

class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)


        ########################
        # Example
        # Suppose we have 3 classifiers, and 6 samples (each row represent maximum prediction for 1 classifier, and the k-th column represents k-th sample)
        # 0.7, 0.5, 0.6, 0.9, 0.5, 0.6
        # 0.8, 0.4, 0.2, 0.8, 0.3, 0.7
        # 0.85, 0.8, 0.5, 0.3, 0.4, 0.9
        # Suppose we force the 3, 2 and 1 samples leave from the 1st, 2nd and 3rd classifier respectively, then:
        # We sort the array for each classifier, and we get: (The index denotes the position)
        # [3, 0, 2, 5, 1, 4]
        # [0, 3, 5, 1, 4, 2]
        # [5, 0, 1, 2, 4, 3]
        # Then we know the 3, 0, 2 should go to exit-0, 5, 1 should go to exit-1 and 4 should go to the 3rd classifier.
        ########################

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k]) # p determines how many percentage samples can exit from exit-0, exit-1, etc.
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i] # We find the sample that have i-th highest prediction value in the k-th classifier
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx] # record the threshold for current stage
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops
