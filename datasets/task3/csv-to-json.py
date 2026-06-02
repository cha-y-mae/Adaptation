import re
import csv
import json
import argparse


def clean(v: str) -> str:
    v = v.strip()
    # Remove trailing escaped-quote artifact (e.g. \n"" or \n") left by
    # spreadsheet exporters in multiline CSV cells.
    v = re.sub(r'\n\s*""+\s*$', '', v)
    return v.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",   type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = [
            {k: clean(v) for k, v in row.items() if k is not None and k != ""}
            for row in csv.DictReader(f)
        ]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Converted {len(rows)} rows → {args.output_json}")


if __name__ == "__main__":
    main()