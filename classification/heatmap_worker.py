import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

CLASSES = ["QCD", "TTBar"]


# -----------------------------
# PATHS
# -----------------------------
def get_model_path(base, cls, epoch):
    return f"{base}/checkpoints/{cls}_epoch_{epoch}.pt"


def get_data_path(base, cls, tag, data_folder):
    return f"{data_folder}/preprocessed_data/{cls}_{tag}_processed.h5"


# -----------------------------
# RUN PROBABILITIES
# -----------------------------
def run_prob(model_path, data_path, output_file, args, sampled=False):

    if os.path.exists(output_file):
        return

    cmd = [
        "python", "compute_probabilities.py",
        "--model_path", model_path,
        "--data_path", data_path,
        "--output_file", output_file,
        "--n_jets", str(args.n_jets),
        "--batch_size", str(args.batch_size),
        "--num_const", str(args.num_const),
    ]

    if sampled:
        cmd += ["--input_key", "sampled_jets"]

    subprocess.run(cmd, check=True)


# -----------------------------
# AUC
# -----------------------------
def compute_auc(folder):

    scores, labels = [], []

    for i, cls in enumerate(CLASSES):
        top = pd.read_csv(f"{folder}/TTBar_{cls}.csv")
        qcd = pd.read_csv(f"{folder}/QCD_{cls}.csv")

        s = top["probs"] - qcd["probs"]

        scores.append(s.values)
        labels.append(np.full(len(s), i))

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


# -----------------------------
# MAIN
# -----------------------------
def main(args):

    tt_epoch = args.tt_epoch
    qcd_epoch = args.qcd_epoch
    tag = args.tag

    run_dir = f"{args.output}/heatmap_{tag}/points"
    os.makedirs(run_dir, exist_ok=True)

    result_file = f"{run_dir}/TTBar_{tt_epoch}_QCD_{qcd_epoch}.txt"

    if os.path.exists(result_file):
        return

    tmp = f"{run_dir}/tmp_{tt_epoch}_{qcd_epoch}"
    os.makedirs(tmp, exist_ok=True)

    # -----------------------------
    # DATA SOURCE
    # -----------------------------
    if tag == "sampled":
        sampled_dir = args.sampled_dir

        data_map = {
            "TTBar": f"{sampled_dir}/epoch_{tt_epoch}/TTBar.h5",
            "QCD": f"{sampled_dir}/epoch_{qcd_epoch}/QCD.h5",
        }

        sampled_flag = True

    else:
        data_map = {
            "TTBar": get_data_path(args.data_folder, "TTBar", tag, args.data_folder),
            "QCD": get_data_path(args.data_folder, "QCD", tag, args.data_folder),
        }

        sampled_flag = False

    # -----------------------------
    # RUN PROBABILITIES
    # -----------------------------
    for model_cls, epoch in [("TTBar", tt_epoch), ("QCD", qcd_epoch)]:
        for data_cls in CLASSES:

            model_path = get_model_path(args.training_folder, model_cls, epoch)
            data_path = data_map[data_cls]

            out = f"{tmp}/{model_cls}_{data_cls}.csv"

            run_prob(model_path, data_path, out, args, sampled_flag)

    # -----------------------------
    # AUC
    # -----------------------------
    auc_score = compute_auc(tmp)

    with open(result_file, "w") as f:
        f.write(str(auc_score))

    print(f"{tag}: {tt_epoch},{qcd_epoch} → {auc_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_folder", required=True)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--tt_epoch", type=int, required=True)
    parser.add_argument("--qcd_epoch", type=int, required=True)

    parser.add_argument("--tag", default="sampled")  # sampled / train / val / test
    parser.add_argument("--sampled_dir", type = str, help = "folder with sampled jets. Structure: sampled_dir/epoch_[EPOCH]/[CLASS].h5")

    parser.add_argument("--n_jets", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_const", type=int, default=100)

    args = parser.parse_args()
    main(args)