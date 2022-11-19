import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Turn output from context selection into squad form")
    parser.add_argument(
        "--context_file",
        type=Path,
        default='./data',
        help="context.json",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default='./data',
        help="test.json",
    )
    parser.add_argument(
        "--input_cs_file",
        type=Path,
        help="Output file from context selection.",
    )
    parser.add_argument(
        "--output_squad_file",
        type=Path,
        help="Input file for QA.",
    )

    args = parser.parse_args()
    return args


def preprocess_squad(context, dataset, cs_outputs):
    """
    e.g. 
    {
        "answers": {
            "answer_start": [1],
            "text": ["This is a test text"]
        },
        "context": "This is a test context.",
        "id": "1",
        "question": "Is this a test?",
        "title": "train test"
    }
    """
    results = []
    for data, cs_output in zip(dataset, cs_outputs):
        result = dict()
        result["id"] = data["id"]
        result["question"] = data["question"]
        result["title"] = ""
        result["answers"] = {
            "answer_start": [0],
            "text": [""]
        }
        result["context"] = context[data["paragraphs"][cs_output["pred"]]]

        results.append(result)

    return results

def loadJsonline(fin):
    ls = []
    for line in fin:
        ls.append(json.loads(line))
    return ls

def dumpJsonline(data_ls, fout):
    for data in data_ls:
        json.dump(data, fout, ensure_ascii=False)
        fout.write('\n')

def main(args):
    with open(args.context_file, 'r', encoding='utf-8') as fin:
        context = json.load(fin)
    
    with open(args.test_file, 'r', encoding='utf-8') as fin:
        dataset = json.load(fin)

    with open(args.input_cs_file, 'r', encoding='utf-8') as cs_file:
        cs_outputs = loadJsonline(cs_file)

    with open(args.output_squad_file, 'w', encoding='utf-8') as squad_file:
        dumpJsonline(preprocess_squad(context, dataset, cs_outputs), squad_file)
    
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)