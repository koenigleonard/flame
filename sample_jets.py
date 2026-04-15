import torch
from argparse import ArgumentParser
import os
import model
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sanitize_state_dict(state_dict):
    # Rmove '_orig_mod.' from the keys in the state dict
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def save_args_to_file(args):
    os.makedirs(args.output_dir, exist_ok=True)

    file_path = os.path.join(args.output_dir, f"{args.name}_config.json")

    # convert Namespace → dict
    args_dict = vars(args)

    with open(file_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Config saved to: {file_path}")

def parse_inputs():

    parser = ArgumentParser()
    #### add arguments here
    parser.add_argument("--model_path", type = str, help = "Path to the model file")
    parser.add_argument("--n_jets", type = int, default = 100, help = "Number of sampled jets")
    parser.add_argument("--max_length", type = int, default = 100, help = "Max length of a generated jet" )
    parser.add_argument("--batch_size", type = int, default = 10, help = "Number of jets sampled together")
    parser.add_argument("--topk", type = int, help = "If set particles get only sampled from the <topk> most probable")
    parser.add_argument("--output_dir", type = str, default = "output/sampled_jets", help = "file name of output h5 file")
    parser.add_argument("--name", type = str, default = "jets", help = "name of the h5 file in which the sampled jets will be saved.")
    parser.add_argument("--temperature", type = float , default = 1.0)

    args = parser.parse_args()
    return args

def sample(sampleModel, device, args, train_args):
    start_time = time.time()

    n_jets = args.n_jets

    progress_bar = tqdm(total=n_jets, desc="Sampling Jets")
    sampled_batches = []
    jets_written = 0

    while jets_written < n_jets:
        current_batch = min(args.batch_size, n_jets - jets_written)
        start_time_batch = time.time()

        jets = sampleModel.sample(
            batch_size=current_batch,
            max_length=args.max_length,
            device=device,
            temperature=args.temperature,
            topk=args.topk,
        )

        jets = jets.cpu().numpy().reshape(current_batch, -1)
        sampled_batches.append(jets)

        jets_written += current_batch
        progress_bar.update(current_batch)

        dt = time.time() - start_time_batch
        speed = current_batch / dt if dt > 0 else float("inf")
        progress_bar.set_postfix({"jets/s": f"{speed:.2f}"})

    progress_bar.close()

    all_jets = np.concatenate(sampled_batches, axis=0).astype(np.int16)

    df = pd.DataFrame(all_jets)

    print(df)

    output_file = os.path.join(args.output_dir, f"{args.name}.h5")
    os.makedirs(args.output_dir, exist_ok=True)

    df.to_hdf(output_file, key = "sampled_jets", mode = "w", complevel=9)

    total_time = time.time() - start_time
    print(f"\nFinished sampling {n_jets} jets")
    print(f"Saved sampled jets to {output_file}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Average speed: {n_jets / total_time:.2f} jets/s")


if __name__ == "__main__":
    args = parse_inputs()

    num_features = 3

    print("Running sampling:")
    print(f"Running on device: {device}")

    print(f"Load model from {args.model_path}")

    checkpoint = torch.load(args.model_path)

    train_args = checkpoint["args"]
    print("Args used for sampling:")
    print("-" * 30)
    for key, value in vars(args).items():
        print(f"{key:<15} | {value}")

    save_args_to_file(args)

    print("Args used for training in the loaded checkpoint:")
    print("-" * 30)
    for key, value in train_args.items():
        print(f"{key:<15} | {value}")

    sampleModel = model.JetTransformer(
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        num_features=num_features,
        num_bin_egdes=train_args["num_phys_bins"],
        dropout=train_args["dropout"],
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        causal_mask = train_args["causal_mask"],
        output_mode=train_args["output_mode"]
    )

    sampleModel.to(device)

    sampleModel.load_state_dict(sanitize_state_dict(checkpoint["model_state_dict"]))

    sampleModel.eval()

    with torch.no_grad():
        sample(sampleModel, device, args, train_args)