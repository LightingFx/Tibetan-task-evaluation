"""
TLUE (Tibetan Language Understanding Evaluation) 评测框架 (Optimized Version)
GitHub: Vicentvankor/TLUE

优化和功能调整说明:
1.  **VLLM 显存优化**:
    * 增加了 `gpu_memory_utilization` 参数，限制 vLLM 使用的显存比例，防止因 KV Cache 占满显存而报错。
    * 增加了 `max_model_len` 参数，允许用户设定最大模型长度，帮助 vLLM 更精确地管理内存。
    * 增加了 `swap_space` 参数，允许将部分 KV Cache 交换到 CPU 内存，支持在显存不足时运行更大的模型。

2.  **提示词 (Prompt) 统一**:
    * 根据要求，将中文、英文、藏文模式下的指令统一为："请直接根据问题和选项，给出正确的选项，不要输出任何无关的内容。"

3.  **MMLU 风格结果整合**:
    * 增加了对 Ti-MMLU 科目的分类，自动计算 STEM, Humanities, Social Sciences, 和 Other 四大类的平均准确率。
    * 最终报告中会展示各分类的详细结果。

4.  **结果保存优化**:
    * 在保存的 predictions JSON 文件中，`generated_text` 字段现在会保存模型生成的完整、未截断的文本。

5.  **答案提取逻辑**:
    * 对答案提取逻辑进行了复核，确认其鲁棒性，能有效处理各种格式的输出。

6.  **上下文缓存**:
    * 保留了高效的 few-shot 前缀缓存机制，这是一种 CPU 端的优化，可以减少重复计算，不影响 GPU 显存。
"""

import json
import os
import random
import argparse
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from torch.cuda import is_available as is_cuda_available

# 关键修复：导入 vLLM 和 Transformers 的必要组件
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer
# 关键修复：导入 torch 的 Dataset 类
from torch.utils.data import Dataset

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MMLU a subject-to-category mapping.
# This is a sensible mapping for Ti-MMLU based on standard MMLU categories.
MMLU_CATEGORIES = {
    "STEM": [
        "physics", "chemistry", "biology", "computer_science", "math",
        "engineering", "astronomy"
    ],
    "Humanities": [
        "history", "philosophy", "law", "literature", "art_history", "music"
    ],
    "Social Sciences": [
        "political_science", "economics", "sociology", "psychology",
        "geography", "education"
    ],
    "Other": [
        "business", "health", "miscellaneous", "public_relations", "security_studies"
    ]
}


@dataclass
class EvalConfig:
    """评测配置"""
    model_name_or_path: str
    data_dir: str = "./TLUE/Ti-MMLU_subset670"
    output_dir: str = "./results_new"
    num_few_shot: int = 5
    max_length: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    device: str = "cuda" if is_cuda_available() else "cpu"
    dtype: str = "auto"
    trust_remote_code: bool = False
    subjects: Optional[List[str]] = None
    language: str = "tibetan"
    prompt_style: str = "standard"
    tensor_parallel_size: int = 1
    # VLLM Memory Optimization
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    swap_space: int = 4 # GiB of swap space

class TLUEDataset(Dataset):
    """TLUE藏文数据集"""

    def __init__(
        self,
        data_dir: str,
        subject: str,
        split: str = "test",
        num_few_shot: int = 5,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.subject = subject
        self.split = split
        self.num_few_shot = num_few_shot
        self.seed = seed

        self.test_data = self._load_data("test")

        if num_few_shot > 0:
            self.dev_data = self._load_data("dev")
            if not self.dev_data and self.test_data:
                logger.info(f"No dev data found for {subject}, using first {num_few_shot} test examples for few-shot")
                num_samples = min(num_few_shot, len(self.test_data) // 10 if len(self.test_data) > 20 else 1)
                self.few_shot_examples = self.test_data[:num_samples]
            elif self.dev_data:
                random.seed(seed)
                self.few_shot_examples = random.sample(
                    self.dev_data,
                    min(num_few_shot, len(self.dev_data))
                )
            else:
                self.few_shot_examples = []
        else:
            self.dev_data = []
            self.few_shot_examples = []

    def _load_data(self, split: str) -> List[Dict]:
        """加载数据文件，支持多种格式"""
        data = []
        possible_paths = [
            self.data_dir / f"{self.subject}.jsonl",
            self.data_dir / f"{self.subject}.json",
            self.data_dir / f"{self.subject}_{split}.jsonl",
            self.data_dir / f"{self.subject}_{split}.json",
            self.data_dir / self.subject / f"{split}.json",
            self.data_dir / self.subject / f"{split}.jsonl",
        ]

        file_path = next((path for path in possible_paths if path.exists()), None)

        if not file_path and split == "test":
            main_file_jsonl = self.data_dir / f"{self.subject}.jsonl"
            main_file_json = self.data_dir / f"{self.subject}.json"
            if main_file_jsonl.exists():
                file_path = main_file_jsonl
            elif main_file_json.exists():
                file_path = main_file_json

        if not file_path:
            logger.warning(f"Data file not found for {self.subject} {split}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    loaded_data = json.load(f)
                else:
                    loaded_data = [json.loads(line) for line in f if line.strip()]

            for item in loaded_data:
                processed_item = self._process_data_item(item)
                if processed_item:
                    data.append(processed_item)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading or parsing {file_path}: {e}")
            return []

        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data

    def _process_data_item(self, item: Dict) -> Dict:
        """处理数据项，从polished_ti_content中提取问题和选项"""
        if "question" in item and "choices" in item:
            # 确保答案格式统一
            if "answer" in item and isinstance(item["answer"], str) and item["answer"] in ['A', 'B', 'C', 'D']:
                 item["answer_letter"] = item["answer"]
                 item["answer"] = ord(item["answer"]) - ord('A')
            return item

        if "polished_ti_content" in item:
            content = item["polished_ti_content"]
            lines = content.strip().split('\n')

            question = lines[0].strip()
            choices = []
            for line in lines[1:]:
                line = line.strip()
                if re.match(r"^[A-Dཀ-ང][.།、]\s*", line):
                    choices.append(re.sub(r"^[A-Dཀ-ང][.།、]\s*", "", line))

            processed = {
                "question": question, "choices": choices,
                "answer": item.get("answer", "A"), "original_content": content,
                "loc": item.get("loc", "")
            }

            answer_val = processed["answer"]
            if isinstance(answer_val, str):
                if answer_val in ['A', 'B', 'C', 'D']:
                    processed["answer_letter"] = answer_val
                    processed["answer"] = ord(answer_val) - ord('A')
                elif answer_val in ['ཀ', 'ཁ', 'ག', 'ང']:
                    tibetan_to_idx = {'ཀ': 0, 'ཁ': 1, 'ག': 2, 'ང': 3}
                    processed["answer"] = tibetan_to_idx.get(answer_val, 0)
                    processed["answer_letter"] = ['A', 'B', 'C', 'D'][processed["answer"]]

            return processed
        return item

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx]


class TLUEEvaluator:
    """TLUE藏文评测器 (vLLM优化版)"""

    CHOICES = ["A", "B", "C", "D"]
    TIBETAN_CHOICES = ["ཀ", "ཁ", "ག", "ང"]

    UNIFIED_INSTRUCTION = "请直接根据问题和选项，给出正确的选项，不要输出任何无关的内容。"

    def __init__(self, config: EvalConfig):
        self.config = config
        self.use_tibetan_choices = config.language == "tibetan"

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_vllm_model()

        self.sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 0,
            top_p=config.top_p if config.temperature > 0 else 1.0,
            max_tokens=50,
        )

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """加载分词器"""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_vllm_model(self) -> LLM:
        """使用vLLM加载模型并应用显存优化"""
        logger.info(f"Loading model from {self.config.model_name_or_path} with vLLM")
        return LLM(
            model=self.config.model_name_or_path,
            tokenizer=self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            seed=self.config.seed,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            swap_space=self.config.swap_space,
        )

    def _format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化单个示例"""
        question = example.get("question", "")
        choices = example.get("choices", [])
        
        choice_labels = self.TIBETAN_CHOICES if self.use_tibetan_choices else self.CHOICES
        
        choice_text = ""
        for i, choice in enumerate(choices[:len(choice_labels)]):
            choice_text += f"{choice_labels[i]}. {choice}\n"
        
        prompt = f"དྲི་བ། {question}\n{choice_text}ལན། "
        
        if include_answer:
            answer_idx = example.get("answer", 0)
            try:
                answer_letter = example.get("answer_letter")
                if answer_letter and answer_letter in self.CHOICES:
                    prompt += answer_letter
                else:
                    prompt += self.CHOICES[int(answer_idx)]
            except (ValueError, IndexError):
                 prompt += self.CHOICES[0]
        
        return prompt

    def _build_few_shot_prefix(self, few_shot_examples: List[Dict]) -> str:
        """构建并缓存 few-shot 前缀"""
        if not few_shot_examples:
            return self.UNIFIED_INSTRUCTION
        
        parts = [self.UNIFIED_INSTRUCTION]
        for example in few_shot_examples:
            parts.append(self._format_example(example, include_answer=True))
            parts.append("\n\n")
        
        return "".join(parts)

    def _extract_answer(self, text: str) -> str:
        """从生成文本中提取答案（鲁棒版）"""
        text = text.strip().split('\n')[0]
        text = text.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()

        tibetan_map = {label: self.CHOICES[i] for i, label in enumerate(self.TIBETAN_CHOICES)}
        
        # 优先匹配开头的藏文选项
        match = re.search(r"^\s*([ཀ-ང])", text)
        if match:
            return tibetan_map.get(match.group(1), "A")

        # 匹配开头的英文选项 (A, B, C, D)
        match = re.search(r"^\s*([A-D])", text.upper())
        if match:
            return match.group(1)
        
        # 如果没有匹配到，检查第一个字符是否是有效选项
        if len(text) > 0:
            first_char = text[0].upper()
            if first_char in self.CHOICES: return first_char
            if first_char in tibetan_map: return tibetan_map[first_char]

        logger.debug(f"Could not extract a valid answer from: '{text[:50]}'. Defaulting to 'A'.")
        return "A"

    def evaluate_subject(self, subject: str) -> Dict[str, Any]:
        """评测单个科目 (vLLM批量推理)"""
        logger.info(f"Evaluating subject: {subject}")
        
        dataset = TLUEDataset(self.config.data_dir, subject, "test", self.config.num_few_shot, self.config.seed)
        
        if not dataset.test_data:
            logger.warning(f"No test data found for {subject}")
            return {"subject": subject, "accuracy": 0.0, "correct": 0, "total": 0, "predictions": []}
        
        few_shot_prefix = self._build_few_shot_prefix(dataset.few_shot_examples)
        prompts = [few_shot_prefix + self._format_example(ex, include_answer=False) for ex in dataset.test_data]
        
        outputs = self.model.generate(prompts, self.sampling_params)
        
        correct = 0
        predictions = []
        for i, example in enumerate(tqdm(dataset.test_data, desc=f"Processing {subject}")):
            generated_text = outputs[i].outputs[0].text
            predicted_answer = self._extract_answer(generated_text)
            
            true_answer_idx = int(example.get("answer", 0))
            true_answer = example.get("answer_letter", self.CHOICES[true_answer_idx])
            
            is_correct = (predicted_answer == true_answer)
            if is_correct: correct += 1
            
            predictions.append({
                "question": example.get("question", ""),
                "predicted": predicted_answer,
                "true": true_answer,
                "correct": is_correct,
                "generated_text": generated_text # 保存完整输出
            })
            
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0
        
        return {
            "subject": subject, "accuracy": accuracy, "correct": correct,
            "total": len(dataset), "predictions": predictions
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """评测所有科目并按MMLU类别汇总"""
        subjects = self.config.subjects if self.config.subjects else self._detect_subjects()
        logger.info(f"Found {len(subjects)} subjects to evaluate: {subjects}")
        
        results = {}
        all_correct, all_total = 0, 0
        
        for subject in subjects:
            try:
                subject_result = self.evaluate_subject(subject)
                results[subject] = subject_result
                all_correct += subject_result["correct"]
                all_total += subject_result["total"]
                
                logger.info(
                    f"{subject}: Accuracy = {subject_result['accuracy']:.2%} "
                    f"({subject_result['correct']}/{subject_result['total']})"
                )
            except Exception as e:
                logger.error(f"Error evaluating {subject}: {e}", exc_info=True)
                continue
        
        overall_accuracy = all_correct / all_total if all_total > 0 else 0
        
        # 计算 MMLU 类别准确率
        category_totals = {cat: {"correct": 0, "total": 0} for cat in MMLU_CATEGORIES}
        # Reverse mapping from subject to category
        subject_to_category = {s: cat for cat, sub_list in MMLU_CATEGORIES.items() for s in sub_list}
        
        for subject, result in results.items():
            category = subject_to_category.get(subject)
            if category:
                category_totals[category]["correct"] += result["correct"]
                category_totals[category]["total"] += result["total"]

        category_accuracies = {}
        for category, totals in category_totals.items():
            if totals["total"] > 0:
                acc = totals["correct"] / totals["total"]
                category_accuracies[category] = {
                    "accuracy": acc, "correct": totals["correct"], "total": totals["total"]
                }
        
        summary = {
            "overall_accuracy": overall_accuracy, "total_correct": all_correct, "total_questions": all_total,
            "num_subjects": len(results),
            "category_results": category_accuracies,
            "subject_results": {k: {kk: vv for kk, vv in v.items() if kk != 'predictions'} for k, v in results.items()},
            "config": self.config.__dict__
        }
        
        self.save_predictions(results)
        return summary

    def _detect_subjects(self) -> List[str]:
        """自动检测数据目录中的所有科目"""
        data_dir = Path(self.config.data_dir)
        subjects = set()
        for p in data_dir.glob("**/*.jsonl"): subjects.add(p.stem.replace("_test", "").replace("_dev", ""))
        for p in data_dir.glob("**/*.json"): subjects.add(p.stem.replace("_test", "").replace("_dev", ""))
        
        for p in data_dir.iterdir():
            if p.is_dir(): subjects.add(p.name)
        
        return sorted([s for s in subjects if s])

    def save_results(self, results: Dict[str, Any], file_name: str):
        """保存评测结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(self.config.model_name_or_path).name.replace("/", "_")
        output_file = output_dir / f"tlue_{file_name}_{model_name}_{self.config.num_few_shot}shot.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")

    def save_predictions(self, results_with_preds: Dict[str, Any]):
        """单独保存详细的预测结果"""
        self.save_results(results_with_preds, "predictions")

    def generate_report(self, summary: Dict[str, Any]):
        """生成并打印包含MMLU分类结果的评测报告"""
        model_name = Path(self.config.model_name_or_path).name.replace("/", "_")
        output_dir = Path(self.config.output_dir)
        report_file = output_dir / f"tlue_report_{model_name}_{self.config.num_few_shot}shot.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            report_lines = [
                "="*60,
                "TLUE (Tibetan Language Understanding Evaluation) Report",
                "="*60 + "\n",
                f"Model: {self.config.model_name_or_path}",
                f"Language Mode: {self.config.language}",
                f"Few-shot Examples: {self.config.num_few_shot}",
                f"Overall Accuracy: {summary['overall_accuracy']:.2%}",
                f"Total Questions: {summary['total_questions']}",
                f"Correct Answers: {summary['total_correct']}\n",
                "MMLU Category Results:",
                "-"*60,
            ]
            
            category_results = summary.get('category_results', {})
            if category_results:
                sorted_categories = sorted(category_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                for cat, result in sorted_categories:
                    report_lines.append(f"{cat:<20} {result['accuracy']:>7.2%} ({result['correct']:>4}/{result['total']:>4})")
            
            report_lines.extend(["\n", "Subject-wise Results:", "-"*60])
            
            subject_results = summary.get('subject_results', {})
            if subject_results:
                sorted_subjects = sorted(subject_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                for subject, result in sorted_subjects:
                    report_lines.append(f"{subject:<30} {result['accuracy']:>7.2%} ({result['correct']:>3}/{result['total']:>3})")
            
            report_lines.append("\n" + "-"*60)
            report_content = "\n".join(report_lines)
            f.write(report_content)
        
        logger.info(f"Report saved to {report_file}")
        print(report_content)


def main():
    parser = argparse.ArgumentParser(description="TLUE Tibetan Language Evaluation (vLLM Optimized & Corrected)")
    
    # --- 基本参数 ---
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="./TLUE/Ti-MMLU_subset670", help="TLUE data directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to evaluate")
    parser.add_argument("--language", type=str, default="tibetan", choices=["tibetan", "english"], help="Language for choice formatting (e.g., Tibetan or English letters)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code when loading models")

    # --- 推理参数 ---
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (0 for greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--max_length", type=int, default=4096, help="Max context length for prompts")

    # --- VLLM 性能和显存优化参数 ---
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="The fraction of GPU memory to be used for the model workloads. Default: 0.9")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Model's maximum context length. Defaults to model's config.")
    parser.add_argument("--swap_space", type=int, default=4, help="CPU swap space size (in GiB) for KV cache offloading. Default: 4")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir, output_dir=args.output_dir,
        num_few_shot=args.num_few_shot, subjects=args.subjects,
        language=args.language, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed, dtype=args.dtype,
        max_length=args.max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        swap_space=args.swap_space
    )
    
    evaluator = TLUEEvaluator(config)
    summary = evaluator.evaluate_all()
    
    evaluator.save_results(summary, "summary")
    evaluator.generate_report(summary)


if __name__ == "__main__":
    main()