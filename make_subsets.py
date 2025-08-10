import argparse
from datasets import load_dataset
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Create local training subsets for R2R router.")
    p.add_argument('--sizes', type=int, nargs='+', required=True,
                   help='List of subset sizes (number of examples).')
    p.add_argument('--output-dir', type=Path, default=Path('data_subsets'),
                   help='Local directory to write Dataset folders.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for shuffle reproducibility.')
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)
    print("Loading full training split from HF…")
    ds = load_dataset("nics-efc/R2R_Router_Training", split="train")
    for k in args.sizes:
        print(f" Shuffling and selecting first {k:,} examples…")
        sub = ds.shuffle(seed=args.seed).select(range(k))
        out_dir = args.output_dir / f"train_{k}_ds"
        print(f" Saving subset (with metadata) to {out_dir}")
        sub.save_to_disk(str(out_dir))
    print("✅ All done! Datasets saved under", args.output_dir)

if __name__ == '__main__':
    main()