## Preface | 前言

This Stanford NLP SST-2 setup is only a lightweight test case for verifying that the training framework, data pipeline, fine-tuning flow, and evaluation scripts work end-to-end. It is not the final task definition of this project and should not be treated as the target production setting.

这个 Stanford NLP SST-2 实验只是一个轻量级测试，用来验证训练框架、数据处理流程、微调链路和评测脚本是否能够端到端正常运行。它不是项目最终任务的正式定义，也不代表最终的生产场景。

## Data processing

Convert SST-2 into an instruction-tuning dataset:

```bash
python test/stanford_nlp_test/toolkit/build_sft_dataset.py \
  --dataset_name stanfordnlp/sst2 \
  --output_path test/stanford_nlp_test/stanfordnlp_sst2_sft
```

If the dataset already exists locally with `save_to_disk()`:

```bash
python test/stanford_nlp_test/toolkit/build_sft_dataset.py \
  --dataset_path /path/to/stanfordnlp_sst2 \
  --output_path test/stanford_nlp_test/stanfordnlp_sst2_sft
```

The converted dataset keeps:

- `prompt`: the instruction only
- `answer`: the target label text
- `input`: `prompt + answer`, compatible with the current training code
- `label`, `label_text`, and `has_label`: the original class target and whether the sample is labeled

Note: the public SST-2 `test` split is unlabeled. During conversion, unlabeled rows keep the `prompt`, set `has_label=false`, and do not append an answer. For metric-based evaluation, use `validation` or another labeled split.

注意：公开版 SST-2 的 `test` split 是没有真实标签的。转换时，这类样本会保留 `prompt`，设置 `has_label=false`，并且不会拼接答案。因此如果你要做准确率等指标评测，应使用 `validation` 或其他带标签的 split。

## Evaluation
Before training, run:
```bash
python test/stanford_nlp_test/evaluate/eval_sft_classifier.py \
    --model_path /home/skl/mkx/model/Qwen2.5-7B-Instruct \
    --split validation
```
look at the output to verify the original model accuracy.

After training, run:

```bash
# --model_path path/to/your/save_checkpoint/model \
# --tokenizer_path path/to/your/save_checkpoint/tokenizer \
# --split your_dataset_split

python test/stanford_nlp_test/evaluate/eval_sft_classifier.py \
  --model_path /mnt/sdb1/mkx/save_checkpoint/Qwen2.5-7B-Instruct/2026-03-12/155633/epoch4-step218/model \
  --tokenizer_path /home/skl/mkx/model/Qwen2.5-7B-Instruct \
  --split validation
```
