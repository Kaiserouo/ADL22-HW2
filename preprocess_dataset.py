import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default='./data',
        help="The directory of dataset.",
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

def main(args):
    with open(args.data_dir / 'context.json', 'r') as fin:
        context = json.load(fin)
    
    for dataset_name in ['train', 'val', 'test']:
        with open(args.data_dir / f'{dataset_name}.json', 'r') as fin:
            dataset = json.load(fin)
        is_test = (dataset_name == 'test')
        with open(args.data_dir / f'swag_{dataset_name}.json', 'w') as fout:
            json.dump(preprocess_swag(context, dataset, is_test), fout)
        with open(args.data_dir / f'squad_{dataset_name}.json', 'w') as fout:
            json.dump(preprocess_swag(context, dataset, is_test), fout)
    
    return

    

if __name__ == '__main__':
    main()