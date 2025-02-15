import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "facebook/opt-125m",
    lora_weights: str = "./lora-alpaca/checkpoint-9375",
    device: str = "cpu"
):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    
    # 预定义一些测试用例
    test_cases = [
        # 知识解释类
        {
            "instruction": "What is Python programming language?",
            "input": ""
        },
        # 代码生成类
        {
            "instruction": "Write a simple function to calculate the sum of two numbers in Python",
            "input": ""
        },
        # 概念分析类
        {
            "instruction": "Compare Python and JavaScript. What are their main differences?",
            "input": ""
        },
        # 问题解决类
        {
            "instruction": "How to handle exceptions in Python?",
            "input": ""
        }
    ]
    
    print("\nStarting model evaluation...")
    for case in test_cases:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{case['instruction']}

### Input:
{case['input']}

### Response:
"""
        print(f"\nTesting: {case['instruction']}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}\n")
        print("-" * 50)


if __name__ == "__main__":
    fire.Fire(main)
