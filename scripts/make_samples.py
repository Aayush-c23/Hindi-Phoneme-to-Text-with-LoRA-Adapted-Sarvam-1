"""Rebuild data_samples/ from the full private JSONL (run locally, not committed).

Usage:
    python scripts/make_samples.py --src_dir /path/to/full/jsonl

Defaults to the sibling data layout used during the original run; override with
--src_dir when the full train.jsonl / val.jsonl live elsewhere.
"""
import argparse
import pathlib

DST = pathlib.Path(__file__).resolve().parents[1] / "data_samples"
DST.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Extract public sample rows")
    parser.add_argument("--src_dir", default=None,
                        help="Folder holding full train.jsonl / val.jsonl")
    args = parser.parse_args()
    src = pathlib.Path(args.src_dir) if args.src_dir else DST
    for src_name, dst_name, k in [
        ("train.jsonl", "train_sample30.jsonl", 30),
        ("val.jsonl", "val_sample10.jsonl", 10),
    ]:
        src_file = src / src_name
        if not src_file.exists():
            print(f"skip {dst_name}: {src_file} not found (pass --src_dir)")
            continue
        lines = open(src_file, encoding="utf-8").readlines()[:k]
        open(DST / dst_name, "w", encoding="utf-8").writelines(lines)
        print(f"{dst_name}: {len(lines)} rows")


if __name__ == "__main__":
    main()
