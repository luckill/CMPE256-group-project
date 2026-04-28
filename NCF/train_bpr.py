import torch
import torch.nn.functional as F
import torch.nn as nn


def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_one_epoch_bpr(model, loader, optimizer, device):
    if len(loader) == 0:
        return 0.0

    model.train()
    total_loss = 0.0

    use_cuda = device.type == "cuda"

    for user, pos_item, neg_item in loader:
        user = user.to(device, non_blocking=use_cuda)
        pos_item = pos_item.to(device, non_blocking=use_cuda)
        neg_item = neg_item.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        pos_scores = model(user, pos_item).view(-1)
        neg_scores = model(user, neg_item).view(-1)

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
