import glob
import json
import random
import time
from argparse import ArgumentParser
from os.path import join
from functools import partial

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer

with_plug_in_data_path = "SFT_data/conversations/conversation_with_plugins"
without_plug_in_data_path = "SFT_data/conversations/conversation_without_plugins"


def load_data(tokenizer, with_plugin=False):
    def _load_data(data_dir):
        print(f"load data files from {data_dir}")
        for file in glob.glob(join(data_dir, "**/*.json"), recursive=True):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                num_turns = data["num_turns"]
                prompt = data["meta_instruction"]
                for i in range(num_turns):
                    turn = data["chat"][f"turn_{i + 1}"]
                    for field in ["Human", "Inner Thoughts", "Commands", "Tool Responses", "MOSS"]:
                        prompt += turn[field]
            tokenized_data = tokenizer(prompt, truncation=True)
            ds.append(tokenized_data)

    ds = []
    _load_data(without_plug_in_data_path)
    if with_plugin:
        _load_data(with_plug_in_data_path)

    ds = sorted(ds, key=lambda x: len(x["input_ids"]))

    print(f"use {len(ds)} examples to quantize model, {with_plugin=}")

    return ds


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--save_and_reload", action="store_true", help="whether save quantized model to disk and reload back")
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument("--use_triton", action="store_true", help="whether use triton to speedup at inference")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="max memory used to load model per gpu")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="max memory used to offload model to cpu")
    parser.add_argument("--quant_batch_size", type=int, default=1, help="examples batch size for quantization")
    parser.add_argument("--with_plugin_data", action="store_true", help="whether use plugin data to quantize model")
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=True
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size),
        max_memory=max_memory
    )

    examples = load_data(tokenizer, with_plugin=args.with_plugin_data)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if not args.quantized_model_dir:
        args.quantized_model_dir = args.pretrained_model_dir

    model.save_quantized(args.quantized_model_dir)
    print(f"quantized model saved to {args.quantized_model_dir}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
