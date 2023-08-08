import torch


def cnt_params(params):
    return sum(p.numel() for p in params if p.requires_grad)


def save_model(state, model_name):
    torch.save(state, model_name, pickle_protocol=4)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
