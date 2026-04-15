import torch
from model import JetTransformer
import os
from helpers_train import *
import pandas as pd
import time
from tqdm import tqdm
from torch.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

num_features = 3 #hard code this because it should never vary

def train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          scheduler,
          args,
          epochs = 20,
          start_epoch = 0,
          start_step = 0,
          best_val_loss = None,
          ):

    #model
    model = torch.compile(model)
    model.train()

    scaler = GradScaler("cuda")

    best_val_loss = best_val_loss if best_val_loss is not None else float('inf')

    training_step = start_step
    epoch_steps = len(train_loader)

    if not args.contin:
        save_checkpoint(model, optimizer, scheduler, epoch=0, step=0, val_loss=best_val_loss, args=args, name = "initial_checkpoint")
        
    if not args.contin and os.path.exists(os.path.join(args.output_dir, f"{args.name}_training_log.csv")):
        print("Warning: training log already exists and will be overwritten.")
        os.remove(os.path.join(args.output_dir, f"{args.name}_training_log.csv"))

    if args.verbose:
        if not args.contin and os.path.exists(os.path.join(args.output_dir, f"{args.name}_batch_loss.csv")):
            os.remove(os.path.join(args.output_dir, f"{args.name}_batch_loss.csv"))

    for epoch in range(start_epoch + 1, epochs + 1):
        total_train_loss = 0

        #stop flag in case early stopping is triggered
        stop = False

        start_time = time.time()

        avg_val_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave = True
        )

        for x in progress_bar:
            training_step += 1

            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                logits = model(x)
                loss = model.loss(logits, x)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            #verbose logging
            if args.verbose:
                ### logging
                log_data = {
                    "step": training_step,
                    "epoch": epoch,
                    "batch_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }

                df = pd.DataFrame([log_data])

                df.to_csv(
                    os.path.join(args.output_path, f"{args.name}_batch_loss.csv"),
                    mode="a",
                    header=not os.path.exists(os.path.join(args.output_path, f"{args.name}_batch_loss.csv")),
                    index=False
                )
                
            #update progess bar
            progress_bar.set_postfix(loss = loss.item())

        if args.recompute_train_loss:
            avg_train_loss = validate(model, train_loader)
        else:
            avg_train_loss = total_train_loss / epoch_steps

        #validate on validation set
        avg_val_loss = validate(model, val_loader)
        model.train()

        #save checkpoints
        if args.checkpoint_mode == "all" or args.checkpoint_mode == "best":
            if avg_val_loss < best_val_loss - args.delta_min:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, training_step, best_val_loss, args, name = f"{args.name}_best")
                print(f"New best checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")

                counter = 0
                stop = False
            else:
                counter += 1

        if args.checkpoint_mode == "all":
            save_checkpoint(model, optimizer, scheduler, epoch, training_step, best_val_loss, args, name = f"{args.name}_epoch_{epoch}")

        if counter >= args.patience:
            stop = True

        #logging
        log_data = {
            "epoch": epoch,
            "step": training_step,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        df = pd.DataFrame([log_data])
        df.to_csv(
            os.path.join(args.output_dir, f"{args.name}_training_log.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(args.output_dir, f"{args.name}_training_log.csv")),
            index=False
        )

        total_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} [Finished] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]} | "
            f"Time: {total_time:.2f} s"
        )

        if stop and args.early_stopping:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {counter} consecutive epochs.")
            break

#for running the validation set
def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc= "Validation", leave = False)

        for x in progress_bar:
            x = x.to(device)

            logits = model(x)
            loss = model.loss(logits, x)

            total_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item())

    avg_loss = total_loss / len(dataloader) #dividing by number of batches
    return avg_loss


if __name__ == '__main__':
    args = parse_inputs()

    print("Running trainings process:")
    print("Trainings parameters:")
    print("-" * 30)
    for key, value in vars(args).items():
        print(f"{key:<15} | {value}")

    print("Running on device:", device)
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    save_args_to_file(args)

    print(args.num_phys_bins)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(args)
    print(f"Train loader batches: {len(train_loader)} | Val loader batches: {len(val_loader)}")
    
    # Build model
    transformer_model = JetTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        num_bin_egdes=args.num_phys_bins,
        dropout=args.dropout,
        add_start=args.add_start,
        add_stop=args.add_stop,
        causal_mask=args.causal_mask,
        output_mode=args.output_mode
    ).to(device)

    # Build optimizer and scheduler
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    scheduler = get_scheduler(optimizer, steps_per_epooch=len(train_loader), args=args)

    if args.contin:
        transformer_model, optimizer, scheduler, start_epoch, start_step, best_val_loss, checkpoint_args = load_checkpoint(transformer_model, optimizer, scheduler, device, args.checkpoint, args)
    else:
        start_epoch = 0
        start_step = 0
        best_val_loss = None

    train(
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        epochs=start_epoch + args.num_epochs,
        start_epoch=start_epoch,
        start_step=start_step,
        best_val_loss=best_val_loss
    )

