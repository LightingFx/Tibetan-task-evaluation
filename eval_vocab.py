import torch
import time
import json
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def calculate_cpt_bpt(model, tokenizer, texts, device, batch_size=8):
    """计算 CpT (Characters per Token) 和 BPT (Bits per Token)"""
    total_tokens = 0
    total_chars = 0
    total_nll = 0.0
    
    model.eval()
    
    # 准备batch
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Calculating BPT/CpT")):
            # 过滤空文本
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue
            
            # 统计字符数
            batch_chars = sum(len(t) for t in batch)
            total_chars += batch_chars
            
            # Tokenize with padding
            encodings = tokenizer(
                batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # 计算每个样本的有效token数和loss
                for i in range(input_ids.size(0)):
                    # 统计有效token数（非padding）
                    valid_tokens = attention_mask[i].sum().item()
                    if valid_tokens <= 1:
                        continue
                    
                    total_tokens += valid_tokens
                    
                    # 计算该样本的NLL
                    # 注意：outputs.loss是整个batch的平均loss
                    # 我们需要单独计算每个样本
                    sample_logits = outputs.logits[i, :-1, :]
                    sample_labels = input_ids[i, 1:]
                    sample_mask = attention_mask[i, 1:]
                    
                    # 计算cross entropy
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(sample_logits, sample_labels)
                    
                    # 只计算非padding位置的loss
                    masked_losses = losses * sample_mask
                    sample_nll = masked_losses.sum().item()
                    
                    if np.isnan(sample_nll) or np.isinf(sample_nll):
                        continue
                    
                    total_nll += sample_nll
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if total_tokens == 0 or total_chars == 0:
        print("Error: No valid data processed!")
        return 0, 0, 0
    
    cpt = total_chars / total_tokens
    avg_nll = total_nll / total_tokens
    bpt = avg_nll / np.log(2) 
    bpc = (total_nll / np.log(2)) / total_chars
    
    print(f"\nProcessed: {total_chars} chars, {total_tokens} tokens")
    
    return cpt, bpt, bpc

def measure_speed(model, tokenizer, prompt, device, n_generate=100):
    """测量推理速度 (Tokens/sec 和 Chars/sec)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 预热
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # 测试
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=n_generate, do_sample=False)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # 计算生成的token数和字符数
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    tokens_per_sec = n_generate / duration
    chars_per_sec = len(generated_text) / duration
    
    return tokens_per_sec, chars_per_sec

def load_texts(file_path, max_samples=None, max_length=2048):
    """从文件加载文本数据"""
    texts = []
    
    if file_path.endswith(".json") or file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    text = json.loads(line).get("text", "")[:max_length]
                    if text.strip():
                        texts.append(text)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = content.split('\n\n')
            if len(paragraphs) < 10:
                paragraphs = content.split('\n')
            
            for i, para in enumerate(paragraphs):
                if max_samples and i >= max_samples:
                    break
                if para.strip():
                    texts.append(para.strip()[:max_length])
    
    return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="藏文词表验证工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer路径")
    parser.add_argument("--data_file", type=str, default="./tibetan_100m.txt", help="评测语料")
    parser.add_argument("--num_samples", type=int, default=5000, help="样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch大小")
    parser.add_argument("--max_length", type=int, default=2048, help="单样本最大长度")
    parser.add_argument("--output_dir", type=str, default="./cpt", help="输出目录")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path, 
        trust_remote_code=True
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )

    # 加载数据
    print(f"Loading data from {args.data_file}...")
    texts = load_texts(args.data_file, max_samples=args.num_samples, max_length=args.max_length)
    
    if not texts:
        print("Error: No valid texts loaded!")
        exit(1)
    
    print(f"Loaded {len(texts)} samples, using batch_size={args.batch_size}")

    # 计算指标
    print("\nCalculating metrics...")
    cpt, bpt, bpc = calculate_cpt_bpt(model, tokenizer, texts, device, batch_size=args.batch_size)
    
    print(f"\nCpT: {cpt:.4f} chars/token")
    print(f"BPT: {bpt:.4f} bits/token")
    print(f"BPC: {bpc:.4f} bits/char")

    # 速度测试
    print("\nTesting speed...")
    test_prompt = "༄༅། །དེང་རབས་ཀྱི་ཚན་རིག་དང་མཐུན་པའི་ཤེས་བྱ་ནི་"
    tokens_per_sec, chars_per_sec = measure_speed(model, tokenizer, test_prompt, device)
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec, {chars_per_sec:.2f} chars/sec")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = Path(args.model_path).name
    output_file = os.path.join(args.output_dir, f"{model_name}.json")
    
    results = {
        "model": args.model_path,
        "num_samples": len(texts),
        "batch_size": args.batch_size,
        "CpT": float(cpt),
        "BPT": float(bpt),
        "BPC": float(bpc),
        "speed_tokens_per_sec": float(tokens_per_sec),
        "speed_chars_per_sec": float(chars_per_sec)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
