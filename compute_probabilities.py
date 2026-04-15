import torch
import model
import time
from tqdm import tqdm
import numpy as np
import dataset
from torch.utils.data import DataLoader
import csv
import os
from argparse import ArgumentParser
from dataset import JetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_inputs():

    parser = ArgumentParser()
    #### add arguments here
    parser.add_argument("--model_path", type = str, help = "Path to the model file")
    parser.add_argument("--data_path", type = str, help = "Path to jet data set of which the probabilities should be computed ")
    parser.add_argument("--n_jets", type = int, default = 100, help = "Number of jets taken from test set")
    parser.add_argument("--num_const", type = int, default = 100, help = "Number of constituents taken from dataset")
    parser.add_argument("--batch_size", type = int, default = 10, help = "Number of jets used in one computation step")
    parser.add_argument("--topk", type = int, help = "If set particles get only sampled from the <topk> most probable")
    parser.add_argument("--output_file", type = str, default = "output/probs/probs.csv", help = "file name of the output csv file")
    parser.add_argument("--temperature", type = float , default = 1.0)
    parser.add_argument("--input_key", type = str, default = "discretized", help = "if the key of table in the h5 is different it can be specified here")
    parser.add_argument("--h5", action="store_true", help = "activate when using sampled jets")
    parser.set_defaults(h5 = False)

    args = parser.parse_args()
    return args


def probabilities(
        sampleModel,
        dataloader,
        device,
        args,
):

    n_jets = args.n_jets

    start_time = time.time()

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    with torch.no_grad():
            progress_bar = tqdm(dataloader, desc= "Computing probabilities", leave = False)

            output_file = open(args.output_file, mode="w", newline="")
            writer = csv.writer(output_file)

            writer.writerow(["probs","multiplicity"])

            for x in progress_bar:
                x = x.to(device)
            
                valid_mask = ~((x == -1).any(dim=-1))   
                seq_lens = valid_mask.sum(dim = 1)

                #create logits from forward pass
                logits = sampleModel.forward(x)
                probs = sampleModel.probability(logits, x, logarithmic = True)
            
                result = torch.stack((probs, seq_lens), dim = 1)
            
                writer.writerows(result.tolist())
                
    total_time = time.time() - start_time
    print(f"\nFinished calculating probabily of {n_jets} jets")
    print(f"Total time: {total_time:.2f} s")
    print(f"Average speed: {n_jets / total_time:.2f} jets/s")    

if __name__ == "__main__":
    args = parse_inputs()

    print(f"Running on device: {device}")

    num_features = 3

    ##load model from file
    print(f"Load model state from:{args.model_path}")
    checkpoint = torch.load(args.model_path)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    train_args = checkpoint["args"]
    print("Model was trained with following arguments:")
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
    
    sampleModel.load_state_dict({
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
                })
    
    sampleModel.eval()
    
    #load datasets
    test_loader = DataLoader(JetDataset(
        h5_file = args.data_path,
        tag = "test",
        num_features=num_features,
        num_phys_bins=train_args["num_phys_bins"],
        num_const=args.num_const,
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        n_jets=args.n_jets,
        key = args.input_key,
        h5 = args.h5
        ),
        batch_size=args.batch_size)

    print(f"Test set size: {len(test_loader)}")

    ## run sampling
    probabilities(sampleModel, test_loader, device, args = args)
    