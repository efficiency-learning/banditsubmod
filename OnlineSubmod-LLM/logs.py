import os
import json
import glob
import argparse
from collections import defaultdict
from statistics import mean
from torch.utils.tensorboard import SummaryWriter


def average_metrics(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["step"]].append(entry)

    averaged = []
    for step, group in sorted(grouped.items()):
        avg_entry = {"step": step}
        keys = group[0].keys() - {"step", "epoch"}
        for k in keys:
            avg_entry[k] = mean(e[k] for e in group)
        avg_entry["epoch"] = group[0].get("epoch", 0)
        averaged.append(avg_entry)
    return averaged

def process_and_log(file_path, output_dir):
    with open(file_path, "r") as f:
        data = json.load(f)

    averaged = average_metrics(data)

    exp_name = os.path.splitext(os.path.basename(file_path))[0]
    writer = SummaryWriter(log_dir=os.path.join(output_dir, exp_name))

    for entry in averaged:
        step = entry["step"]
        for key, value in entry.items():
            if key not in {"step", "epoch"}:
                writer.add_scalar(key, value, step)

    writer.close()
    print(f"Logged: {exp_name}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.json"))
    for file_path in files:
        process_and_log(file_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with JSON log files")
    parser.add_argument("--output_dir", type=str, default="tb_logs", help="Directory for TensorBoard logs")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
