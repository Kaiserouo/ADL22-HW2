import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Graph QA output")
    parser.add_argument(
        "--input_result_file",
        type=Path,
        help="The all_result.json file generated with QA model.",
    )
    parser.add_argument(
        "--train_loss_remove_point",
        type=int,
        nargs='+',
        help="Remove those step numbers from train loss graph."
    )

    args = parser.parse_args()
    return args

def main(args):
    # get results
    with args.input_result_file.open('r') as fin:
        result = json.load(fin)
    result = result["metric_steps"]
    
    # graph EM
    x = [int(k) for k in result.keys()]
    y = [result[str(i)]["exact_match"] for i in x]
    plt.plot(x, y, color="blue", label="Valid EM")
    plt.legend(loc='upper left')
    plt.show()

    # graph loss
    x = [int(k) for k in result.keys() if int(k) not in args.train_loss_remove_point]
    print(x)
    print(args.train_loss_remove_point)
    y = [result[str(i)]["train_loss"] for i in x]
    plt.plot(x, y, color="orange", label="Train loss")
    plt.legend(loc='upper right')
    plt.show()
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)