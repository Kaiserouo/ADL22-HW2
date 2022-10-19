import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--input_data_dir",
        type=Path,
        default='./data',
        help="The directory of dataset.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=Path,
        default='./data',
        help="The directory of output dataset.",
    )

    args = parser.parse_args()
    return args

def preprocess_swag(context, dataset, is_test = False):
    """
    When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label" if "label" in column_names else "labels"
    """
    results = []
    for data in dataset:
        result = dict()
        result["id"] = data["id"]
        result["sent1"] = ""    # I don't know what to put
        result["sent2"] = data["question"]
        for i, para_id in enumerate(data["paragraphs"]):
            result[f"ending{i}"] = context[para_id]
        if not is_test:
            result["label"] = data["paragraphs"].index(data["relevant"])

        results.append(result)
    
    return results

def preprocess_squad(context, dataset, is_test = False):
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
    for data in dataset:
        result = dict()
        result["id"] = data["id"]
        result["question"] = data["question"]
        result["title"] = ""
        if not is_test:
            result["answers"] = {
                "answer_start": [data["answer"]["start"]],
                "text": [data["answer"]["text"]]
            }
            result["context"] = context[data["relevant"]]

        results.append(result)
    
    return results

def dumpJson(data_ls, fout):
    for data in data_ls:
        json.dump(data, fout, ensure_ascii=False)
        fout.write('\n')

def main(args):
    with open(args.input_data_dir / 'context.json', 'r', encoding='utf-8') as fin:
        context = json.load(fin)
    
    for dataset_name in ['train', 'valid', 'test']:
        with open(args.input_data_dir / f'{dataset_name}.json', 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
        is_test = (dataset_name == 'test')
        with open(args.output_data_dir / f'swag_{dataset_name}.json', 'w', encoding='utf-8') as fout:
            dumpJson(preprocess_swag(context, dataset, is_test), fout)
        with open(args.output_data_dir / f'squad_{dataset_name}.json', 'w', encoding='utf-8') as fout:
            dumpJson(preprocess_squad(context, dataset, is_test), fout)
    
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)