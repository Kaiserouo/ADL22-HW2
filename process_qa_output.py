import json
import csv
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Turn output from qa into CSV format")
    parser.add_argument(
        "--input_qa_file",
        type=Path,
        help="Output file from qa.",
    )
    parser.add_argument(
        "--output_csv_file",
        type=Path,
        help="Predict file.",
    )

    args = parser.parse_args()
    return args



def main(args):
    with open(args.output_csv_file, 'w', encoding='utf-8') as fout:
        with open(args.input_qa_file, 'r', encoding='utf-8') as fin:
            qa = json.load(fin)
            csvwriter = csv.writer(fout, delimiter=",")
            csvwriter.writerow(['id', 'answer'])
            for id, ans in qa.items():
                csvwriter.writerow([id, ans])
    
if __name__ == '__main__':
    args = parse_args()
    main(args)