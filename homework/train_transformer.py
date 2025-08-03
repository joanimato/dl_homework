"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

from datetime import datetime
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import TransformerPlanner, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "transformer_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    #Use SGD and L1
    loss_func = torch.nn.SmoothL1Loss()
    # optimizer = ...
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    global_step = 0
    # metrics = {"train_acc": [], "val_acc": []}
    train_accuracy = PlannerMetric()
    val_accuracy = PlannerMetric()
    
    # metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):

        train_accuracy.reset()
        val_accuracy.reset()

        model.train()

        running_loss = 0.0

        for datum in train_data:
            # 'track_left', 'track_right', 'waypoints', 'waypoints_mask'
            track_left = datum['track_left'].to(device)
            track_right = datum['track_right'].to(device)
            waypoints = datum['waypoints'].to(device)
            waypoints_mask = datum['waypoints_mask'].to(device)

            optimizer.zero_grad()

            pred = model(track_left, track_right)
            # pred = pred.view(-1,3,2)
            train_loss = loss_func(pred, waypoints)

            train_accuracy.add(pred, waypoints, waypoints_mask)
            
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for datum in val_data:
                # 'track_left', 'track_right', 'waypoints', 'waypoints_mask'
                track_left = datum['track_left'].to(device)
                track_right = datum['track_right'].to(device)
                waypoints = datum['waypoints'].to(device)
                waypoints_mask = datum['waypoints_mask'].to(device)

                pred = model(track_left, track_right)
                # pred = pred.view(-1,3,2)
            
                # ADD metric
                val_accuracy.add(pred, waypoints, waypoints_mask)

        # log average train and val accuracy to tensorboard
        train_acc = train_accuracy.compute()
        val_acc = val_accuracy.compute()

            # "l1_error": float(l1_error),
            # "longitudinal_error": float(longitudinal_error),
            # "lateral_error": float(lateral_error),
            # "num_samples": self.total,
        logger.add_scalar("train l1_error", train_acc["l1_error"], epoch)
        logger.add_scalar("train longitudinal_error", train_acc["longitudinal_error"], epoch)
        logger.add_scalar("train lateral_error", train_acc["lateral_error"], epoch)
        logger.add_scalar("val l1_error", val_acc["l1_error"], epoch)
        logger.add_scalar("val longitudinal_error", val_acc["longitudinal_error"], epoch)
        logger.add_scalar("val lateral_error", val_acc["lateral_error"], epoch)

        epoch_loss = running_loss / batch_size

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"Epoch Loss={epoch_loss:.4f} "
                f"train_l1={train_acc['l1_error']:.4f} "
                f"val_long_error={val_acc['longitudinal_error']:.4f} "
                f"val_lat_error={val_acc['lateral_error']:.4f} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
