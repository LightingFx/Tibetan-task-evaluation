import json
import random
import re
import argparse
import os  # 【新增】用于处理路径
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- 1. Prompt 模板定义 ---
PROMPT_TEMPLATES = {
    "direct": {
        "zh": {
            "system": "以下是数学应用题回答，请只给出最终的数字答案，不要包含任何单位、解释或多余的文字。",
            "example_separator": "\n\n",
            "example_format": "问题：{question}\n答案：{answer}",
            "final_prompt": "问题：{question}\n答案：",
        },
        "bo": {
            "system": "འདི་ནི་བརྩི་གཞི་འགའ་ཡིན། ལན་དུ་མཐའ་མའི་ཨང་གྲངས་ཁོ་ན་བྲིས་རོགས། འགྲེལ་བརྗོད་དང་ཚིག་ལྷག་གང་ཡང་མི་དགོས།",
            "example_separator": "\n\n",
            "example_format": "དྲི་བ། {question}\nལན། {answer}",
            "final_prompt": "དྲི་བ། {question}\nལན།",
        }
    },
    "cot": {
        "zh": {
            "system": "以下是数学应用题回答，前面是给出的例子，你只需要回答最后的问题。请详细写出你的推理步骤，并确保将最终答案以 {answer:数字} 的格式放在结尾。",
            "example_separator": "\n\n",
            "example_format": "问题：{question}\n答案：{reasoning}\n最终答案是{{answer:{answer}}}",
            "final_prompt": "问题：{question}\n答案：",
        },
        "bo": {
            "system": "འདི་ནི་བརྩི་གཞི་འགའ་ཡིན།  གནད་དོན་ཐག་གཅོད་བྱེད་པའི་བརྒྱུད་རིམ་བསམ་གཞིགས་བྱས་ནས་བྲིས་རོགས།  མཐའ་མའི་ལན་དེ་ངེས་པར་དུ་ {answer:ཨང་གྲངས།} ཟེར་བའི་རྣམ་པའི་ནང་འཇོག་དགོས།",
            "example_separator": "\n\n",
            "example_format": "དྲི་བ། {question}\nལན། {reasoning}\nམཐའ་མཇུག་གི་དྲིས་ལན་ནི་{{answer:{answer}}}",
            "final_prompt": "དྲི་བ། {question}\nལན།",
        }
    }
}

def load_dataset(
    filepath: str,
    num_samples: int,
    n_shot: int,
    seed: int
) -> (List[Dict], List[Dict]):
    """
    加载数据集，并根据'split'字段严格划分训练集和测试集。
    """
    random.seed(seed)
    
    if not os.path.exists(filepath):
        print(f"错误: 数据集文件 '{filepath}' 未找到。")
        exit(1)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取JSON文件: {e}")
        exit(1)

    train_pool = [item for item in dataset if item.get('split') == 'train']
    test_pool = [item for item in dataset if item.get('split') == 'test']

    if len(train_pool) < n_shot:
        print(f"错误: 训练集样本数 ({len(train_pool)}) 不足以支持 {n_shot}-shot。")
        exit(1)
    if len(test_pool) < num_samples:
        print(f"错误: 测试集样本数 ({len(test_pool)}) 不足以支持 {num_samples} 个评估样本。")
        exit(1)

    few_shot_examples = random.sample(train_pool, n_shot)
    test_samples = random.sample(test_pool, num_samples)
    
    print(f"已加载数据集：从 'train' 集中随机抽取 {len(few_shot_examples)} 个样本作为 Few-shot 示例。")
    print(f"从 'test' 集中随机抽取 {len(test_samples)} 个样本用于评估。")
    return few_shot_examples, test_samples


def build_prompt(
    sample: Dict,
    examples: List[Dict],
    template: Dict,
    mode: str
) -> str:
    """
    根据选择的模式 (direct/cot) 构建 prompt。
    """
    prompt = template["system"] + template["example_separator"]
    
    if examples:
        example_texts = []
        for ex in examples:
            if mode == 'direct':
                example_texts.append(template["example_format"].format(
                    question=ex["question_bo"], 
                    answer=ex["answer_only"]
                ))
            else: # cot mode
                example_texts.append(template["example_format"].format(
                    question=ex["question_bo"],
                    reasoning=ex["answer_bo"],
                    answer=ex["answer_only"]
                ))
        prompt += template["example_separator"].join(example_texts)
        prompt += template["example_separator"]

    prompt += template["final_prompt"].format(question=sample["question_bo"])
    return prompt


def extract_answer_number(text: str, mode: str) -> Optional[str]:
    """
    从模型输出中提取答案。
    """
    if mode == 'cot':
        # 寻找 {answer: xxx} 或 {answer：xxx} (全角冒号) 格式
        match = re.search(r'\{answer[:：\s]*([-+]?\d*\.?\d+)\s*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).strip('. ')
    
    # Direct 模式兜底，或 CoT 格式失败时的备选方案
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1].strip('. ') 
    return None


def evaluate(args):
    """
    主评估函数。
    """
    # 【新增】从路径中提取干净的模型名称 (去除路径和尾部斜杠)
    model_name = os.path.basename(os.path.normpath(args.model_path))

    print("=" * 20)
    print("开始进行 GSM8K-BO 数据集评估")
    print(f"模型名称: {model_name}")  # 打印提取出的模型名
    print(f"提示词模式: {args.prompt_mode.upper()}")
    print(f"Shot 数: {args.n_shot}")
    print(f"评估样本数: {args.num_samples}")
    print("=" * 20)

    # --- 2. 加载模型 ---
    print("\n[1/4] 正在加载 vLLM 模型...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=20960,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    
    print("\n[2/4] 正在加载并处理数据集...")
    few_shot_examples, test_samples = load_dataset(args.dataset_path, args.num_samples, args.n_shot, args.random_seed)
    prompt_template = PROMPT_TEMPLATES[args.prompt_mode][args.prompt_lang]

    # --- 3. 构建 Prompts ---
    print("\n[3/4] 正在构建所有评估用的 Prompts...")
    prompts = [build_prompt(sample, few_shot_examples, prompt_template, args.prompt_mode) for sample in test_samples]
    
    # --- 4. 模型推理 ---
    max_tokens = 1024 if args.prompt_mode == 'cot' else 50
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    print(f"\n[4/4] 开始使用 vLLM 进行批量推理 (共 {len(prompts)} 条)...")
    outputs = llm.generate(prompts, sampling_params)

    correct_count = 0
    results = []

    for i, output in enumerate(tqdm(outputs, desc="评估进度")):
        model_output = output.outputs[0].text.strip()
        model_answer = extract_answer_number(model_output, args.prompt_mode)
        
        ground_truth = str(test_samples[i]["answer_only"])
        
        is_correct = (model_answer == ground_truth)
        if is_correct:
            correct_count += 1
            
        results.append({
            "question": test_samples[i]["question_bo"],
            "ground_truth": ground_truth,
            "model_output": model_output,
            "extracted_answer": model_answer,
            "correct": is_correct,
            "prompt": prompts[i],
        })

    # --- 5. 输出结果 ---
    accuracy = (correct_count / len(test_samples)) * 100
    
    print("\n" + "=" * 20)
    print("评估完成！")
    print(f"模型: {model_name}")
    print(f"准确率: {accuracy:.2f}% ({correct_count}/{len(test_samples)})")
    print("=" * 20)

    # 【优化】动态生成结果文件名
    # 格式: gsm8k_results_{model_name}_{prompt-mode}_{n-shot}shot.json
    output_filename = f"gsm8k_results_{model_name}_{args.prompt_mode}_{args.n_shot}shot.json"
    
    # 确保保存路径有效 (可选: 这里保存在当前目录)
    output_path = os.path.join(os.getcwd(), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"详细评估结果已保存至: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 评估藏文大模型在 GSM8K-BO 上的表现")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM 兼容的藏文大模型路径")
    parser.add_argument("--dataset-path", type=str, default="./eval_data/gsm8k_bo.json", help="评估数据集文件路径")
    parser.add_argument("--prompt-mode", type=str, default="cot", choices=["direct", "cot"], help="选择提示词模式")
    parser.add_argument("--n-shot", type=int, default=2, help="Few-shot 学习的示例数量")
    parser.add_argument("--num-samples", type=int, default=500, help="用于评估的测试样本数量")
    parser.add_argument("--prompt-lang", type=str, default="bo", choices=["bo", "zh"], help="选择提示词语言")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM 张量并行大小")
    
    args = parser.parse_args()
    evaluate(args)
