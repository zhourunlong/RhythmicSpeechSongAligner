import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataset, loss_func, args):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    losses = []

    with tqdm(dataloader, desc="Evaluating") as pbar:
        for inputs in pbar:
            outputs = model(inputs["sen1"], inputs["sen2"])
            loss = loss_func(outputs, inputs["scores"].cuda())
            losses.append(loss.item())

            pbar.set_description("Loss: %0.8f" % (np.mean(losses)))

    return np.mean(losses)
