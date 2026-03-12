import argparse
from typing import Dict

from datasets import DatasetDict, load_dataset, load_from_disk


LABEL_MAP = {
    0: "negative",
    1: "positive",
}


PROMPT_TEMPLATE = """User:
You are a sentiment classification assistant.
Read the input sentence and decide whether its sentiment is positive or negative.
Return only one word as the final answer: positive or negative.

Sentence:
{sentence}

Assistant:
"""


def load_source_dataset(dataset_path: str, dataset_name: str | None) -> DatasetDict:
    if dataset_path:
        return load_from_disk(dataset_path)
    if dataset_name:
        return load_dataset(dataset_name)
    raise ValueError("Either --dataset_path or --dataset_name must be provided.")


def convert_example(example: Dict) -> Dict:
    label = int(example["label"])
    prompt = PROMPT_TEMPLATE.format(sentence=example["sentence"].strip())
    has_label = label in LABEL_MAP
    answer = LABEL_MAP[label] if has_label else ""
    return {
        "idx": example.get("idx", -1),
        "label": label,
        "label_text": answer if has_label else "unknown",
        "has_label": has_label,
        "sentence": example["sentence"],
        "prompt": prompt,
        "answer": answer,
        "input": prompt + answer if has_label else prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an instruction-tuning dataset from SST-2.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="test/stanford_nlp_test/data/stanfordnlp_sst2",
        help="Path to a local Hugging Face dataset saved with save_to_disk().",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Hugging Face dataset name. Example: stanfordnlp/sst2.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test/stanford_nlp_test/data/stanfordnlp_sst2_sft",
        help="Output directory for the converted dataset.",
    )
    args = parser.parse_args()

    dataset = load_source_dataset(args.dataset_path, args.dataset_name or None)
    converted = DatasetDict()

    for split_name, split_dataset in dataset.items():
        converted[split_name] = split_dataset.map(
            convert_example,
            remove_columns=split_dataset.column_names,
            desc=f"Converting {split_name}",
        )

    converted.save_to_disk(args.output_path)
    print(f"Saved converted dataset to: {args.output_path}")
    for split_name, split_dataset in converted.items():
        print(f"{split_name}: {len(split_dataset)} rows, columns={split_dataset.column_names}")
        print(split_dataset[0])


if __name__ == "__main__":
    main()
