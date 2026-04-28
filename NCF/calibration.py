import torch
from torch import nn, optim


class ScoreCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, scores):
        scale = torch.exp(self.log_scale)
        return torch.sigmoid(scale * scores + self.bias)


class CalibratedModel(nn.Module):
    def __init__(self, base_model, calibrator):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator

    def forward(self, user_id, item_id):
        scores = self.base_model(user_id, item_id).float()
        return self.calibrator(scores)


def fit_score_calibrator(model, loader, device):
    model.eval()
    use_cuda = device.type == "cuda"
    scores_all = []
    ratings_all = []

    with torch.inference_mode():
        for users, items, ratings in loader:
            users = users.to(device, non_blocking=use_cuda)
            items = items.to(device, non_blocking=use_cuda)

            scores = model(users, items).float().detach().cpu()
            scores_all.append(scores)
            ratings_all.append(ratings.float().detach().cpu())

    if not scores_all:
        calibrator = ScoreCalibrator()
        return calibrator, 1.0, 0.0

    scores_tensor = torch.cat(scores_all)
    ratings_tensor = torch.cat(ratings_all)

    calibrator = ScoreCalibrator()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(
        calibrator.parameters(),
        lr=0.5,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        preds = calibrator(scores_tensor)
        loss = criterion(preds, ratings_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    scale = float(torch.exp(calibrator.log_scale).item())
    bias = float(calibrator.bias.item())
    calibrator = calibrator.to(device)
    return calibrator, scale, bias