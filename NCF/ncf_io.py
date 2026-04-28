import torch

from NCF.model import neural_cllaborative_filtering


def maybe_compile_model(model):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            return model
    return model

def save_model_checkpoint(model, config, path):
    base_model = unwrap_model(model)
    checkpoint = {
        "model_state_dict": base_model.state_dict(),
        "config": dict(config),
        "embedding_dimension": int(config["embedding_dimension"]),
    }
    torch.save(checkpoint, path)


def load_model_checkpoint(path, num_users, num_items, device):
    checkpoint = torch.load(path, map_location=device)

    model = neural_cllaborative_filtering(
        num_users=num_users,
        num_items=num_items,
        embedding_dimension=int(checkpoint["embedding_dimension"]),
        use_sigmoid=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model