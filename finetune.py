import os
import sys
from typing import List
import logging

import torch
import transformers
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import requests
import json
import psutil
import gc

"""
Unused imports:
import torch.nn as 
import bitsandbytes as bnb
"""

from peft import (
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.prompter import Prompter

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Peft version: {peft.__version__}")
print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# 1. API 设置
API_KEY = os.getenv("HF_API_KEY")  # Get API key from environment variable
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

def query_deepseek(prompt, api_key=None):
    """
    Call DeepSeek API to generate text
    Args:
      prompt: Input text prompt
      api_key: Optional API key, will use environment variable if not provided
    Returns:
      Generated text or None if error
    """
    if not api_key:
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            logging.error("No API key provided. Set HF_API_KEY environment variable")
            return None
            
    headers = {"Authorization": f"Bearer {api_key}"}
    api_url = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logging.error(f"API Error: {e}")
        return None

# 2. 加载本地模型
def load_local_model():
    """加载本地 LLaMA 模型"""
    model_name = "your_local_llama_path"  # 替换为你的本地模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# 3. 准备训练数据
def prepare_training_data():
    """从 jsonl 文件准备训练数据"""
    training_data = []
    with open('data/crafting_instruction.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            prompt = f"How do I craft {item['instruction']}?"
            training_data.append({
                'prompt': prompt,
                'target': item['output']
            })
    return training_data

# 4. 知识蒸馏训练
def train_with_distillation(student_model, tokenizer, training_data, num_epochs=3):
    """使用知识蒸馏训练本地模型"""
    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    
    student_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for item in training_data:
            # 获取教师模型输出
            teacher_output = query_deepseek(item['prompt'], API_KEY)
            if not teacher_output:
                continue
                
            # 准备输入
            input_ids = tokenizer(
                item['prompt'], 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids
            
            # 准备标签
            labels = tokenizer(
                teacher_output[0]['generated_text'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids
            
            # 训练步骤
            optimizer.zero_grad()
            outputs = student_model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

def prepare_dataset(data_path):
    """
    Load and prepare dataset from jsonl file for training
    Args:
      data_path: Path to the jsonl data file
    Returns:
      Processed dataset ready for training
    """
    # Load dataset from jsonl
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Format the text in a chat-like format
            text = f"### Instruction: {item['instruction']}\n"
            text += f"### Input: {item['input']}\n"
            text += f"### Response: {item['output']}\n"
            dataset.append({"text": text})
    
    from datasets import Dataset
    dataset = Dataset.from_list(dataset)
    
    return dataset

def preprocess_function(examples):
    """
    Preprocess the examples for training
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # Set up the labels for training
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def train(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    data_path: str = "data/crafting_instruction.jsonl",
    output_dir: str = "./lora-crafting",
    api_key: str = None,
    batch_size: int = 4,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='right')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Preparing dataset...")
    # 将 preprocess_function 移到这里，这样它可以访问 tokenizer
    def preprocess_function(examples):
        text = examples["text"]
        text = f"{text}{tokenizer.eos_token}"
        
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
    
    raw_dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
            raw_dataset.append({"text": text})
    
    from datasets import Dataset
    dataset = Dataset.from_list(raw_dataset)
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        num_proc=1
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        push_to_hub=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        torch_compile=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def get_memory_info():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory used: {memory_info.rss / 1024 / 1024:.2f} MB"

def test_setup():
    print("\n=== 环境检查 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Initial {get_memory_info()}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n=== 使用设备: {device} ===")
    
    try:
        print("\n=== 开始加载模型 ===")
        # 使用更小的模型进行测试
        model_name = "facebook/opt-125m"  # 只有125M参数的小模型
        
        print("1. 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer 加载成功 - {get_memory_info()}")
        
        print("\n2. 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print(f"模型加载成功 - {get_memory_info()}")
        
        print("\n3. 移动模型到设备...")
        model = model.to(device)
        print(f"模型已移动到 {device} - {get_memory_info()}")
        
        print("\n=== 测试简单推理 ===")
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=10,
                num_return_sequences=1
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: {input_text}")
        print(f"输出: {result}")
        print(f"Final {get_memory_info()}")
        
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return False

def clear_memory():
  """Clear GPU memory before training"""
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()
  torch.cuda.empty_cache()  # For GPU users
  gc.collect()  # Clear CPU memory
  
def get_memory_status():
  """Get current memory usage status"""
  memory_info = {}
  
  if torch.backends.mps.is_available():
    memory_info['mps_allocated'] = torch.mps.current_allocated_memory() / 1024**3
    memory_info['mps_max'] = 9.07  # Your MPS max memory
    
  process = psutil.Process()
  memory_info['ram_used'] = process.memory_info().rss / 1024**3
  memory_info['ram_percent'] = psutil.virtual_memory().percent
  
  return memory_info

if __name__ == "__main__":
    # Clear memory before starting
    clear_memory()
    initial_memory = get_memory_status()
    print("\nInitial memory status:")
    for k, v in initial_memory.items():
        print(f"  {k}: {v:.2f} GB")
    
    training_config = {
        "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "data_path": "data/crafting_instruction.jsonl",
        "output_dir": "./lora-crafting",
        "batch_size": 2,  # Reduced from 4
        "micro_batch_size": 1,  # Reduced from 2
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "max_length": 256,  # Added max length control
    }
    
    try:
        print("\n=== Starting Crafting Instructions Training ===\n")
        
        # 打印数据集大小
        with open(training_config["data_path"], 'r') as f:
            num_examples = sum(1 for _ in f)
        print(f"Found {num_examples} training examples\n")
        
        print("Training configuration:")
        for k, v in training_config.items():
            print(f"  {k}: {v}")
            
        success = train(**training_config)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
