from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from dataset import JetDataset
from torch.utils.data import DataLoader
import math
import model
import torch.optim.lr_scheduler
import json

def parse_inputs():
    parser = ArgumentParser(description="Train the JetTransformer model.")
    
    #data paths
    parser.add_argument('--train_file', type=str, default='data/train.h5', help='Path to the training dataset file (HDF5 format).')
    parser.add_argument('--val_file', type=str, default='data/val.h5', help='Path to the validation dataset file (HDF5 format).')
    parser.add_argument('--input_key', type=str, default='discretized', help='Key in the HDF5 file to load the data from.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model checkpoints and logs.')
    
    #dataset parameters
    parser.add_argument('--name', type = str, default = "model", help = "Name for the model and training run. Used for saving checkpoints and logs.")
    parser.add_argument('--n_jets', type =int , default = None, help = "Number of jets to load from the dataset. If None, loads all jets.")
    parser.add_argument('--n_jets_val', type =int , default = None, help = "Number of jets to load from the validation dataset. If None, loads all jets.")
    parser.add_argument('--num_phys_bins', type=int, nargs=3, default=(40, 30, 30), help='Number of bins for pt, eta, and phi discretization.')
    parser.add_argument('--add_start', action='store_true', help='Whether to add a start token to the input sequences.')
    parser.set_defaults(add_start=True)
    parser.add_argument('--add_stop', action='store_true', help='Whether to add a stop token to the input sequences.')
    parser.set_defaults(add_stop=True)
    parser.add_argument('--num_const', type=int, default=50, help='Maximum number of constituents per jet.')
    parser.add_argument('--no_shuffle', action='store_true', help='Whether to disable shuffling of the training data.')
    parser.set_defaults(no_shuffle=False)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')

    #model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for the model.')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer layers.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--causal_mask', action='store_true', help='Whether to use causal masking in the transformer.')
    parser.add_argument('--output_mode', type=str, default='linear', choices=['linear'], help='Output mode for the model.')
    parser.set_defaults(causal_mask=True)
    parser.add_argument('--positional_encoding', action='store_true', help='Wheter to use positional encoding in the embedding')
    parser.set_defaults(positional_encoding=False)

    #training parameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')
    parser.add_argument('--batch_size_val', type=int, default=100, help='Batch size for validation.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--scheduler', type = str, default = "constant", choices = ["constant", "cosine", "lambda", "cosine_restarts", "exp", "warmup_cosine"], help = "Learning rate scheduler to use during training.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer.')
        ##-- continue mode
    parser.add_argument('--contin', action='store_true', help = "Whether to continue training from a checkpoint. If set, the model will be loaded from the specified checkpoint and training will continue from there.")
    parser.set_defaults(contin=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to load the model from if --contin is set. The checkpoint should be a .pt file containing the model state dict and optimizer state dict.')
    parser.add_argument('--reset_optimizer', action='store_true', help='Whether to reset the optimizer state when continuing training from a checkpoint. If set, the model will be loaded from the checkpoint but the optimizer will be re-initialized, effectively resetting the learning rate schedule and momentum.')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--reset_scheduler', action='store_true', help='Whether to reset the learning rate scheduler state when continuing training from a checkpoint. If set, the model will be loaded from the checkpoint but the learning rate scheduler will be re-initialized, effectively resetting the learning rate schedule.')
    parser.set_defaults(reset_scheduler=False)
    parser.add_argument('--new_lr', type=float, default=None, help='New learning rate to use when continuing training from a checkpoint. If set, this learning rate will be used to override the learning rate loaded from the checkpoint.')
        ##----
    parser.add_argument('--checkpoint_mode', type = str, default = "all", choices = ["all", "best"], help = "Whether to save checkpoints for all epochs or only for the best epoch based on validation loss.")
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped (early stopping).')
    parser.add_argument('--early_stopping', action='store_true', help='Whether to use early stopping based on validation loss.')
    parser.set_defaults(early_stopping=False)
    parser.add_argument('--delta_min', type=float, default=1e-4, help='Minimum change in validation loss to qualify as an improvement for early stopping.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer to use for training.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print detailed training progress and metrics during training.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--restart_period', type=int, default=10, help='Number of epochs between restarts for the cosine annealing scheduler with restarts.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Value by which the learning rate will be multiplied at each epoch for the exponential scheduler.')
    parser.add_argument('--recompute_train_loss' , action='store_true', help='Whether to recompute the training loss on the entire training set at the end of each epoch.')

    args = parser.parse_args()
    return args

def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        # warmup phase
        if step < warmup_steps:
            return step / float(warmup_steps)

        # cosine decay phase
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def constant_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0, total_steps)

def get_scheduler(optimizer, steps_per_epooch, args):

    if args.scheduler == "warmup_cosine":
        scheduler = warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(0.1*steps_per_epooch*args.num_epochs),
            total_steps=steps_per_epooch*args.num_epochs
        )
        print("Using cosine scheduler with warmup.")
    elif args.scheduler == "constant":
        scheduler = constant_scheduler(
            optimizer,
            total_steps=steps_per_epooch*args.num_epochs
        )
        print("Using constant scheduler.")
    elif args.scheduler == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = args.restart_period * steps_per_epooch + 1,
            eta_min = args.lr_min,
        )
        print(f"Using cosine restart scheduler with a T_0 = {args.restart_period} and eta_min = {args.lr_min}")
    elif args.scheduler == "exp":

        gamma = args.gamma**(1/steps_per_epooch)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = gamma
        )
        print(f"Using exponential scheduler with gamma = {args.gamma}.")
    elif args.scheduler == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max= steps_per_epooch*args.num_epochs,
            eta_min= args.lr_min)
        
        print(f"Using cosine decay scheduler with eta_min = {args.lr_min}")

    else:
        scheduler = constant_scheduler(
            optimizer,
            total_steps=steps_per_epooch*args.num_epochs,
        )
        print("Using constant scheduler.")

    return scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, args, name = "latest"):

    path = f"{args.output_dir}/checkpoints"
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "args": vars(args)
    }

    torch.save(checkpoint, f"{path}/{name}.pt")

    print(f"Checkpoint saved: {path}/{name}.pt")

def load_checkpoint(model, optimizer, scheduler, device, checkpoint_path, args):

    print("Loading checkpoint from:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Args used for training in the loaded checkpoint:")
    print("-" * 30)
    for key, value in checkpoint["args"].items():
        print(f"{key:<15} | {value}")

    model.load_state_dict(sanitize_state_dict(checkpoint["model_state_dict"]))

    #load optimizer state only if reset_optimizer is not set and the checkpoint contains an optimizer state dict
    if not args.reset_optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(sanitize_state_dict(checkpoint["optimizer_state_dict"]))
        print("Optimizer state loaded from checkpoint.")
    else:
        print("Optimizer state NOT loaded from checkpoint. Initializing new optimizer.")
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=args.new_lr or args.lr, 
                                      weight_decay=args.weight_decay) if args.optimizer == "adamw" else torch.optim.Adam(model.parameters(), 
                                                                                                                         lr=args.new_lr or args.lr, 
                                                                                                                         weight_decay=args.weight_decay)

    #load scheduler state only if reset_scheduler is not set, the checkpoint contains a scheduler state dict, and a scheduler is provided
    if scheduler is not None and not args.reset_scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(sanitize_state_dict(checkpoint["scheduler_state_dict"]))
        print("Scheduler state loaded from checkpoint.")
    else:
        print("Scheduler state NOT loaded from checkpoint. Initializing new scheduler.")
        scheduler = get_scheduler(optimizer, steps_per_epooch=math.ceil(args.n_jets / args.batch_size), args=args)

    #set new lr if specified
    if args.new_lr is not None:

        print(f"Overriding learning rate to {args.new_lr}.")

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.new_lr

        if hasattr(scheduler, 'base_lrs'):
            scheduler.base_lrs = [args.new_lr for _ in scheduler.base_lrs]

    start_epoch = checkpoint["epoch"] + 1
    start_step = checkpoint["step"]

    val_loss = checkpoint.get("val_loss", None)

    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}, step {start_step}. Validation loss at checkpoint: {val_loss}")

    return model, optimizer, scheduler, start_epoch, start_step, val_loss, checkpoint["args"]

def sanitize_state_dict(state_dict):
    # Rmove '_orig_mod.' from the keys in the state dict
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def save_args_to_file(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.contin:
        counter = 1
        file_path = os.path.join(args.output_dir, f"{args.name}_config_contin_{counter}.json")
        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(args.output_dir, f"{args.name}_config_contin_{counter}.json")
    else:
        file_path = os.path.join(args.output_dir, f"{args.name}_config.json")

    # convert Namespace → dict
    args_dict = vars(args)

    with open(file_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Config saved to: {file_path}")

def build_dataloaders(args):

    train_dataset = JetDataset(
        h5_file=args.train_file,
        key=args.input_key,
        n_jets=args.n_jets,
        num_const=args.num_const,
        num_phys_bins=args.num_phys_bins,
        add_start=args.add_start,
        add_stop=args.add_stop,
        tag = "train"
    )

    val_dataset = JetDataset(
        h5_file=args.val_file,
        key=args.input_key,
        n_jets=args.n_jets_val,
        num_const=args.num_const,
        num_phys_bins=args.num_phys_bins,
        add_start=args.add_start,
        add_stop=args.add_stop,
        tag= "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return train_loader, val_loader