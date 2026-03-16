# Standalone debug for `col_train_lora.py` before `model.train()`

这个目录是独立调试副本，只复现 `training/col_train_lora.py` 在 `model.train()` 之前的关键初始化流程：

- 解析参数
- 加载 tokenizer
- 加载 model config
- 构建模型
- 自动推断 LoRA target modules
- 注入 LoRA
- 输出可调试信息
- 可选执行一次 dummy forward

这里不会加载数据集，也不会依赖仓库内任何 Python 模块。把这个目录单独拷走后，只要环境里有 `torch`、`transformers`、`peft`，就可以直接运行。

## Files

- `standalone_debug.py`: 主脚本
- `original_flow_debug.py`: 更接近 `training/col_train_lora.py` 原始控制流的调试脚本，只把 dataset/dataloader 段落去掉
- `requirements.txt`: 这个独立目录所需的最小依赖

## Install

```bash
pip install -r test/debug_col_train_lora_pre/requirements.txt
```

## Example

```bash
python test/debug_col_train_lora_pre/standalone_debug.py \
  --model_path /path/to/model \
  --init_mode config \
  --print_modules 20
```

如果你要的是“基本按原脚本顺序跑，只跳过 dataloader”：

```bash
torchrun --nproc_per_node=1 test/debug_col_train_lora_pre/original_flow_debug.py \
  --model_path /path/to/model \
  --plugin hybrid_parallel \
  --print_limit 20
```

如果你想顺便看一次前向输出：

```bash
python test/debug_col_train_lora_pre/standalone_debug.py \
  --model_path /path/to/model \
  --init_mode pretrained \
  --run_dummy_forward \
  --dummy_text "hello world"
```

## Debug suggestion

你可以直接在 `standalone_debug.py` 里下断点，重点位置是：

- `build_model_and_lora()`
- `print_debug_snapshot()`
- `run_dummy_forward()`

其中 `main()` 里 `build_model_and_lora()` 返回之后，对应的就是原脚本 `model.train()` 之前的状态。

如果你要和原代码逐行对照，优先看 `original_flow_debug.py`。它保留了这段代码的原始顺序，包括：

- `colossalai.launch_from_torch({})`
- `plugin` / `booster` 初始化
- `get_parallel_ranks()`
- `AutoTokenizer.from_pretrained(...)`
- `AutoConfig.from_pretrained(...)`
- `with init_ctx: ... booster.enable_lora(...)`

只把 dataset / shuffle / prepare_dataloader 三段替换成了空占位。
