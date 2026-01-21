import os
import re
import json
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from vllm import LLM, SamplingParams

# Optional LoRA
try:
    from vllm.lora.request import LoRARequest
    HAS_LORA = True
except Exception:
    HAS_LORA = False

# -------------------------
# Logging setup
# -------------------------
def setup_logging(level: str = "WARNING"):
    level = (level or "WARNING").upper()
    os.environ.setdefault("VLLM_LOGGING_LEVEL", level)  # respected by recent vLLM
    log_level = getattr(logging, level, logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")
    for name in ["vllm", "transformers", "httpx", "urllib3", "fastapi", "uvicorn"]:
        # 设置这些库的日志级别为 WARNING，以减少非进度条的输出
        # 但 vLLM 本身对 VLLM_LOGGING_LEVEL 的尊重使得这里通常不用改
        logging.getLogger(name).setLevel(max(logging.WARNING, log_level))
    if level == "ERROR":
        hf_logging.set_verbosity_error()
    elif level == "WARNING":
        hf_logging.set_verbosity_warning()
    elif level == "INFO":
        hf_logging.set_verbosity_info()
    else:
        hf_logging.set_verbosity_debug()

# -------------------------
# Qwen thinking clean-ups
# -------------------------
THINK_TAG_PATTERNS = [
    r"<think>.*?</think>",
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
    r"<\|thought\|>.*?</\|thought\|>",
]

def strip_thinking_blocks(text: str) -> str:
    out = text or ""
    for pat in THINK_TAG_PATTERNS:
        out = re.sub(pat, "", out, flags=re.DOTALL | re.IGNORECASE)
    return out.strip()

# -------------------------
# Data loading
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# -------------------------
# Prompt (Tibetan-focused, bilingual guardrails)
# -------------------------
PROMPT_SYSTEM = (
    "You are an expert evaluator for Tibetan multiple-choice reading comprehension. "
    "Read the passage and the question carefully, reason step by step if needed, "
    "then choose EXACTLY ONE option.\n"
    "CRITICAL: End your reply with a single line in the exact format {{answer: X}}, "
    "where X is one of A, B, C, D. Do not add any text after the closing braces."
)

# Tibetan-facing user template; preserves structure stability
def build_user_content_bod(passage: str, question: str, A: str, B: str, C: str, D: str) -> str:
    return (
        "འགྲེལ་བཤད། གཤམ་གསལ་གྱི་དོན་ཚན་དང་དྲི་བ་ལ་དག་དོན་དང་བརྗོད་པའི་གནས་ཚུལ་གཞིར་བཟུང་ནས་"
        "དྲི་ཚན་གྱི་དམིགས་བསལ་གཅིག་གདམ་རོགས། "
        "མཇུག་ཏུ་ངེས་པར་དུ {{answer: X}} རྣམས་ཀྱི་རྣམ་པའི་རྣམ་གྲངས་X ∈ {A,B,C,D} གཅིག་གིས་སྟོན་རོགས།\n\n"
        "【དོན་ཚན】\n"
        f"{passage}\n\n"
        "【དྲི་བ】\n"
        f"{question}\n\n"
        "【གདམ་བ】\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n\n"
        "གནད་དོན་འགྲེལ་བཤད་བྱས་རྗེས་{{answer: X}} ཡིག་ཚུལ་དང་དབྱེ་བ་དེའི་གྲངས་ལས་བསྡུས་"
        "གཞན་དག་སྐབས་མར་འདོན་མ་བྱེད།"
    )

def build_messages(passage: str, question: str, answers: List[str]) -> List[Dict[str, str]]:
    A, B, C, D = answers
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": build_user_content_bod(passage, question, A, B, C, D)},
    ]

# -------------------------
# Engine & generation
# -------------------------
@dataclass
class Engine:
    llm: LLM
    tokenizer: AutoTokenizer

def load_engine(model_name: str,
                gpu_memory_utilization: float = 0.9,
                max_model_len: Optional[int] = None) -> Engine:
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
        enable_lora=True
    )
    tok = llm.get_tokenizer()
    return Engine(llm=llm, tokenizer=tok)

def render_prompts(tokenizer: AutoTokenizer,
                   messages_batch: List[List[Dict[str, str]]],
                   disable_thinking: bool = True) -> List[str]:
    prompts = []
    for msgs in messages_batch:
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=(not disable_thinking)  # disable for stability
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)
    return prompts

def batched(iterable, bs):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == bs:
            yield buf; buf = []
    if buf:
        yield buf

# -------------------------
# Strict answer parsing
# -------------------------
ANSWER_RE = re.compile(r"\{\{\s*answer\s*[:：]\s*([A-D1-4])\s*\}\}", re.IGNORECASE)

def parse_answer_tag(text: str) -> Optional[str]:
    """Return 'A'|'B'|'C'|'D' if found; otherwise None."""
    if not text:
        return None
    txt = strip_thinking_blocks(text)
    m = ANSWER_RE.search(txt)
    if m:
        g = m.group(1).upper()
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        return mapping.get(g, g)
    # fallback heuristics: last uppercase letter A-D at end of line
    tail = re.findall(r"[^A-D]([A-D])[\)\.\s]*$", txt, re.IGNORECASE)
    if tail:
        return tail[-1].upper()
    return None

# -------------------------
# Sampling helpers
# -------------------------
def build_indices(n: int, max_samples: Optional[int], mode: str, seed: int) -> List[int]:
    if not max_samples or max_samples >= n:
        return list(range(n))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=max_samples, replace=False).tolist()
    return list(range(max_samples))

# -------------------------
# Utility for dynamic naming
# -------------------------
def extract_model_name(path: str) -> str:
    """Extracts the final directory/file name from a path."""
    if not path:
        return "unknown_model"
    # os.path.basename handles both '/path/to/model' and '/path/to/model/'
    return os.path.basename(path.rstrip(os.path.sep))


# -------------------------
# Main eval
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="./eval_data/bele_bod.jsonl", help="Path to Belebele bod_Tibt jsonl")
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--lora", type=str, default=None, help="Optional LoRA adapter dir")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=900)
    ap.add_argument("--sample-mode", type=str, choices=["head","random"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./bele_out")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-level", type=str, choices=["ERROR","WARNING","INFO","DEBUG"], default="WARNING")
    args = ap.parse_args()

    setup_logging(args.log_level)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_jsonl(args.jsonl)
    n_all = len(rows)
    idx = build_indices(n_all, args.max_samples, args.sample_mode, args.seed)
    rows = [rows[i] for i in idx]
    logging.info("Loaded %d items (subset of %d).", len(rows), n_all)

    engine = load_engine(args.base_model)
    llm = engine.llm
    tokenizer = engine.tokenizer

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[],  # we rely on parsing final tag; allow free reasoning
    )

    lora_req = None
    if args.lora and HAS_LORA:
        lora_req = LoRARequest("belebele-adapter", 1, args.lora)

    # Prepare prompts
    def rec_to_msgs(r):
        passage = r.get("flores_passage","").strip()
        question = r.get("question","").strip()
        answers = [r.get(f"mc_answer{i}", "").strip() for i in range(1,5)]
        return build_messages(passage, question, answers)

    preds = []
    golds = []
    qnums = []
    outs = []

    # 使用 tqdm 封装批处理循环，显示总进度
    for chunk in tqdm(list(batched(rows, args.batch_size)), desc="Infer", disable=args.no_progress):
        msgs_batch = [rec_to_msgs(r) for r in chunk]
        prompts = render_prompts(tokenizer, msgs_batch, disable_thinking=True)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)

        for r, out in zip(chunk, outputs):
            text = out.outputs[0].text if out.outputs else ""
            text = strip_thinking_blocks(text)
            pred = parse_answer_tag(text)
            preds.append(pred)
            gold_num = str(r.get("correct_answer_num", ""))
            gold = {"1":"A","2":"B","3":"C","4":"D"}.get(gold_num, None)
            golds.append(gold)
            qnums.append(r.get("question_number"))
            outs.append({
                "question_number": r.get("question_number"),
                "dialect": r.get("dialect"),
                "pred": pred,
                "gold": gold,
                "raw_output": text
            })

    # Accuracy
    correct = sum(1 for p,g in zip(preds,golds) if (p is not None and g is not None and p==g))
    total = len(golds) # 实际处理的样本数
    acc = correct / total if total else 0.0

    # --- 动态命名和保存 ---
    model_short_name = extract_model_name(args.base_model)
    # 使用实际处理的样本数 (total) 来命名文件
    sample_tag = f"N{total}" 
    
    pred_filename = f"{model_short_name}_{sample_tag}_predictions.jsonl"
    summary_filename = f"{model_short_name}_{sample_tag}_summary.json"
    
    # Save predictions
    out_pred_path = os.path.join(args.out_dir, pred_filename)
    with open(out_pred_path, "w", encoding="utf-8") as f:
        for o in outs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            
    # Save summary
    out_summary_path = os.path.join(args.out_dir, summary_filename)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "lora": args.lora,
            "num_items": total,
            "accuracy": acc,
            "max_samples": args.max_samples,
            "sample_mode": args.sample_mode,
            "seed": args.seed
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== Belebele (bod_Tibt) ===")
    print(f"Items: {total} | Accuracy: {acc:.4f}  (correct: {correct})")
    print(f"Predictions saved to: {out_pred_path}")
    print(f"Summary saved to: {out_summary_path}")

if __name__ == "__main__":
    main()
