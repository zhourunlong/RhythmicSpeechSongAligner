import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os, sys, logging, time
from tqdm import tqdm
from dataset import SemanticMatchingDataset
from model import SemanticMatchingModel
import numpy as np
from evaluate import evaluate
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=100, type=int)
    parser.add_argument("--num-epoch-ft", default=50, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logdir = "Experiment-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    print("Experiment dir : {}".format(logdir))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    shutil.copy("dataset.py", os.path.join(logdir, "code"))
    shutil.copy("evaluate.py", os.path.join(logdir, "code"))
    shutil.copy("model.py", os.path.join(logdir, "code"))
    shutil.copy("train.py", os.path.join(logdir, "code"))

    train_set = SemanticMatchingDataset("train")
    valid_set = SemanticMatchingDataset("valid")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)

    model = SemanticMatchingModel()
    model.cuda()
    
    loss_func = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0.1*args.lr)

    loss_eval_best = 1e9

    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="Training") as pbar:
            losses = []
            for inputs in pbar:
                optimizer.zero_grad()
                outputs = model(inputs["sen1"], inputs["sen2"])
                loss = loss_func(outputs, inputs["scores"].cuda())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                pbar.set_description("Epoch: %d, Loss: %0.8f, LR: %0.6f" % (epoch, np.mean(losses), optimizer.param_groups[0]['lr']))

            logging.info("Epoch: %d, Loss: %0.8f, LR: %0.6f" % (epoch, np.mean(losses), optimizer.param_groups[0]['lr']))
        
        loss_eval = evaluate(model, valid_set, loss_func, args)
        logging.info("Valid loss: %0.8f" % (loss_eval))

        #state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        state_dict = {"model": model.state_dict()}

        savepath = os.path.join(logdir, "models/%d.pt" % (epoch))
        if (epoch + 1) % 10 == 0:
            torch.save(state_dict, savepath)
            logging.info("Saving to {}".format(savepath))

        if loss_eval < loss_eval_best:
            loss_eval_best = loss_eval
            savepath = os.path.join(logdir, "models/best.pt")
            torch.save(state_dict, savepath)
            logging.info("Best eval loss! Saving to {}".format(savepath))
        
        scheduler.step()

    # fine-tuning
    state_dict = torch.load(os.path.join(logdir, "models/best.pt"))
    model.load_state_dict(state_dict["model"])
    model.unfreeze_bert()
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.num_epoch_ft):
        model.train()
        with tqdm(train_loader, desc="Fine-tuning") as pbar:
            losses = []
            for inputs in pbar:
                ft_optimizer.zero_grad()
                outputs = model(inputs["sen1"], inputs["sen2"])
                loss = loss_func(outputs, inputs["scores"].cuda())
                losses.append(loss.item())
                loss.backward()
                ft_optimizer.step()

                pbar.set_description("Epoch: %d, Loss: %0.8f, LR: %0.6f" % (epoch, np.mean(losses), ft_optimizer.param_groups[0]['lr']))

            logging.info("Epoch: %d, Loss: %0.8f, LR: %0.6f" % (epoch, np.mean(losses), ft_optimizer.param_groups[0]['lr']))
        
        loss_eval = evaluate(model, valid_set, loss_func, args)
        logging.info("Valid loss: %0.8f" % (loss_eval))

        #state_dict = {"model": model.state_dict(), "optimizer": ft_optimizer.state_dict()}
        state_dict = {"model": model.state_dict()}

        savepath = os.path.join(logdir, "models/ft_%d.pt" % (epoch))
        if (epoch + 1) % 10 == 0:
            torch.save(state_dict, savepath)
            logging.info("Saving to {}".format(savepath))

        if loss_eval < loss_eval_best:
            loss_eval_best = loss_eval
            savepath = os.path.join(logdir, "models/ft_best.pt")
            torch.save(state_dict, savepath)
            logging.info("Best eval loss! Saving to {}".format(savepath))
        
        scheduler.step()
