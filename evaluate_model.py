import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import argparse

def load_model(base_model, lora_weights):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text=""):
    # 使用更简单的提示模板
    prompt = f"{instruction}"  # 直接使用指令
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,        # 减少最大长度
        temperature=0.7,           # 增加温度使输出更多样
        top_p=0.9,
        top_k=50,                 # 添加 top_k 采样
        num_beams=1,              # 不使用束搜索
        do_sample=True,
        repetition_penalty=1.5,    # 增加重复惩罚
        no_repeat_ngram_size=3,   # 避免 n-gram 重复
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

def evaluate_model(
    base_model: str = "facebook/opt-125m",
    lora_weights: str = "./lora-crafting-v2",
    device: str = "cpu"
):
    print("\n=== 开始模型评估 ===\n")
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model, tokenizer = load_model(base_model, lora_weights)
    
    test_cases = [
        {
            "instruction": "请解释如何制作一个工作台？",
            "input": "",
            "category": "crafting",
            "difficulty": "easy"
        },
        {
            "instruction": "制作铁镐需要什么材料？",
            "input": "",
            "category": "materials",
            "difficulty": "medium"
        },
        {
            "instruction": "如何制作一个箱子来存储物品？",
            "input": "",
            "category": "crafting",
            "difficulty": "easy"
        }
    ]
    
    results = []
    for case in test_cases:
        print(f"\n测试类别: {case['category']}")
        print(f"难度级别: {case['difficulty']}")
        print(f"测试指令: {case['instruction']}\n")
        
        response = generate_response(model, tokenizer, case['instruction'], case.get('input', ''))
        print(f"生成的回答:\n{response}\n")
        
        # 自动评分，不再需要用户输入
        score = 1  # 默认最低分
        if len(response) > 0 and not response.startswith(case['instruction']):
            if len(response.split()) > 10:  # 如果回答超过10个词
                score = 3
            if len(response.split()) > 20:  # 如果回答超过20个词
                score = 4
        
        print(f"自动评分: {score}/5\n")
        print("-" * 50 + "\n")
        
        results.append({
            'instruction': case['instruction'],
            'response': response,
            'score': score
        })
    
    # 打印总结
    print("\n=== 评估总结 ===")
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"平均分数: {avg_score:.2f}/5")
    
    return results

def analyze_results(results):
    categories = {}
    difficulties = {}
    
    for case in results:
        cat = case['category']
        diff = case['difficulty']
        score = case['score']
        
        if cat not in categories:
            categories[cat] = []
        if diff not in difficulties:
            difficulties[diff] = []
            
        categories[cat].append(score)
        difficulties[diff].append(score)
    
    print("\n=== 评估结果分析 ===\n")
    print("按类别平均分:")
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"{cat}: {avg:.2f}/5")
    
    print("\n按难度平均分:")
    for diff, scores in difficulties.items():
        avg = sum(scores) / len(scores)
        print(f"{diff}: {avg:.2f}/5")
    
    overall_score = sum(case['score'] for case in results) / len(results)
    print(f"\n总体评分: {overall_score:.2f}/5")
    
    print("\n=== 改进建议 ===")
    if overall_score < 3:
        print("模型需要进一步微调:")
        print("1. 增加训练数据量")
        print("2. 调整学习率")
        print("3. 增加训练轮数")
    elif overall_score < 4:
        print("模型表现一般，建议:")
        print("1. 针对性补充训练数据")
        print("2. 微调特定类别的任务")
    else:
        print("模型表现良好，可以:")
        print("1. 扩展到更复杂的任务")
        print("2. 尝试特定领域的优化")

def check_training_info(lora_weights: str):
    """检查训练信息"""
    print(f"\n=== 训练信息分析 ===")
    
    # 1. 检查目录结构
    checkpoints = [d for d in os.listdir(lora_weights) if d.startswith('checkpoint-')]
    print(f"\n发现的检查点:")
    for cp in sorted(checkpoints):
        print(f"- {cp}")
    
    # 2. 检查最新的checkpoint
    latest_cp = sorted(checkpoints)[-1] if checkpoints else None
    if latest_cp:
        latest_path = f"{lora_weights}/{latest_cp}"
        print(f"\n最新检查点: {latest_cp}")
        
        # 检查配置文件
        config_path = f"{latest_path}/adapter_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                print("\nLoRA 配置:")
                print(f"- LoRA 秩 (r): {config.get('r', 'N/A')}")
                print(f"- LoRA alpha: {config.get('lora_alpha', 'N/A')}")
                print(f"- 目标模块: {config.get('target_modules', 'N/A')}")
                print(f"- 任务类型: {config.get('task_type', 'N/A')}")
        
        # 检查模型文件
        model_path = f"{latest_path}/adapter_model.bin"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n模型文件大小: {size:.2f} MB")
    
    print("\n=== 训练状态评估 ===")
    print("1. 检查点数量:", len(checkpoints))
    print("2. 训练完整性:", "完整" if latest_cp else "不完整")
    
    # 3. 给出建议
    print("\n=== 建议 ===")
    if not checkpoints:
        print("- 训练可能未成功完成，建议检查训练过程")
    else:
        print("- 建议进行模型效果测试")
        print("- 可以比较不同检查点的效果")
        print("- 考虑是否需要更多训练轮数")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default="facebook/opt-125m")
    parser.add_argument('--lora_weights', default="./lora-crafting-v2")
    parser.add_argument('--device', default="cpu")
    args = parser.parse_args()
    
    evaluate_model(args.base_model, args.lora_weights, args.device)

if __name__ == "__main__":
    main() 