import argparse
import re
from typing import Dict, List

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL_TEXTS = ("negative", "positive")


def parse_prediction(text: str) -> str:
    lowered = text.strip().lower()
    for label in LABEL_TEXTS:
        if re.search(rf"\b{label}\b", lowered):
            return label
    return "unknown"


def build_generation_kwargs(tokenizer) -> Dict:
    kwargs = {
        "max_new_tokens": 8,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    return kwargs


def evaluate_batch(model, tokenizer, prompts: List[str], device: torch.device) -> List[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, 4096),
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        generated = model.generate(**encoded, **build_generation_kwargs(tokenizer))

    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    predictions = []
    for i, seq in enumerate(generated):
        continuation = seq[int(prompt_lengths[i]) :]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        predictions.append(parse_prediction(text))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on an instruction-style binary classification dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the model tokenizer.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="test/stanford_nlp_test/data/stanfordnlp_sst2_sft",
        help="Path to the processed evaluation dataset.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit the number of evaluated samples.")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)[args.split]
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if "has_label" in dataset.column_names:
        labeled_indices = [idx for idx, flag in enumerate(dataset["has_label"]) if flag]
        if not labeled_indices:
            raise ValueError(
                f"Split '{args.split}' does not contain ground-truth labels. "
                "SST-2 test is unlabeled, so evaluate on train/validation instead."
            )
        if len(labeled_indices) != len(dataset):
            print(
                f"Warning: split '{args.split}' contains unlabeled rows. "
                f"Evaluating only {len(labeled_indices)} labeled samples."
            )
            dataset = dataset.select(labeled_indices)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    device = next(model.parameters()).device

    golds: List[str] = []
    preds: List[str] = []
    unknown_count = 0

    for start in tqdm(
        range(0, len(dataset), args.batch_size),
        desc=f"Evaluating {args.split}",
        unit="batch",
    ):
        batch = dataset[start : start + args.batch_size]
        batch_preds = evaluate_batch(model, tokenizer, batch["prompt"], device)
        preds.extend(batch_preds)
        golds.extend(batch["label_text"])
        unknown_count += sum(1 for pred in batch_preds if pred == "unknown")

    total = len(golds)
    correct = sum(int(pred == gold) for pred, gold in zip(preds, golds))
    accuracy = correct / total if total else 0.0

    per_label = {}
    for label in LABEL_TEXTS:
        tp = sum(1 for pred, gold in zip(preds, golds) if pred == label and gold == label)
        fp = sum(1 for pred, gold in zip(preds, golds) if pred == label and gold != label)
        fn = sum(1 for pred, gold in zip(preds, golds) if pred != label and gold == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for gold in golds if gold == label),
        }

    print(f"Split: {args.split}")
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Unknown predictions: {unknown_count}")
    for label, metrics in per_label.items():
        print(
            f"{label}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )

    print("\nExamples:")
    for i in range(min(5, total)):
        print(f"[{i}] gold={golds[i]}, pred={preds[i]}")
        print(dataset[i]["sentence"])
        print("-" * 60)


if __name__ == "__main__":
    main()
