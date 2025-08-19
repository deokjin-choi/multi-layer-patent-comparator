#!/usr/bin/env python
import os, json, csv, pickle
import torch, torch.nn.functional as F
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NLP_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 경로
BASE_MODEL_PATH     = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
LORA_OUTPUT_DIR     = os.path.join(NLP_PAPER_DIR, "data/peft/lora_output/mistral_lora_quant")
VALID_JSONL         = os.path.join(NLP_PAPER_DIR, "data/peft/fine_tuned_valid.jsonl")
JUSTIF_DIR          = os.path.join(NLP_PAPER_DIR, "data/peft/justifications")
PROBS_DIR           = os.path.join(NLP_PAPER_DIR, "data/peft/token_probs")

TOP_K = 100

def build_prompt_shared(entry):
    input_ = entry["input"]
    return (
        f"Below is a comparison between two patents: Our patent and the Competitor's patent.\n\n"
        f"Functional Purpose:\n"
        f"- Winner: {input_['fp_winner']}\n"
        f"- Reason: {input_['fp_reason']}\n\n"
        f"Technical Uniqueness:\n"
        f"- Winner: {input_['tu_winner']}\n"
        f"- Reason: {input_['tu_reason']}\n\n"
        f"Strategic Value:\n"
        f"- Winner: {input_['sv_winner']}\n"
        f"- Reason: {input_['sv_reason']}\n\n"
        f"overall_winner: {input_['overall_winner']}\n\n"
        f"Write an overall judgement (1–2 sentences) justifying the overall winner.\n"
        f"Do not start with sentences like 'The overall winner is...'.\n"
        f"This explanation must reflect the actual aspect(s) that contributed to the win "
        f"(e.g., functional purpose, technical uniqueness, or strategic value).\n"
    )

def load_valid_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_outputs_csv(idx2text, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index", "overall_reason"])
        for i, txt in idx2text.items():
            w.writerow([i, txt])

def save_probs_pkl(idx2probs, path_pkl):
    os.makedirs(os.path.dirname(path_pkl), exist_ok=True)
    with open(path_pkl, "wb") as f:
        pickle.dump(idx2probs, f)

def run_inference(model, tokenizer, entries, top_k=TOP_K):
    model.eval()
    idx2text, idx2probs = {}, {}
    for i, entry in enumerate(tqdm(entries)):
        prompt = build_prompt_shared(entry)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True
            )
        # 토큰 확률 top-k
        token_probs = []
        for score in gen.scores:
            probs = F.softmax(score, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            token_dict = { int(idx): float(p) for idx, p in zip(topk_idx[0], topk_probs[0]) }
            token_probs.append(token_dict)
        idx2probs[i] = token_probs

        gen_tokens = gen.sequences[0][input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        idx2text[i] = gen_text
    return idx2text, idx2probs

def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map="auto", trust_remote_code=True,
        quantization_config=bnb, local_files_only=True
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok

def load_finetuned_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map="auto", trust_remote_code=True,
        quantization_config=bnb, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, LORA_OUTPUT_DIR, is_trainable=False, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok

def main():
    entries = load_valid_entries(VALID_JSONL)

    # Base
    base_model, tokenizer = load_base_model()
    base_txt, base_probs = run_inference(base_model, tokenizer, entries)
    save_outputs_csv(base_txt, os.path.join(JUSTIF_DIR, "base_outputs.csv"))
    save_probs_pkl(base_probs, os.path.join(PROBS_DIR, "base_probs.pkl"))
    del base_model; torch.cuda.empty_cache()
    

    # Finetuned
    ft_model, tokenizer = load_finetuned_model()
    ft_txt, ft_probs = run_inference(ft_model, tokenizer, entries)
    save_outputs_csv(ft_txt, os.path.join(JUSTIF_DIR, "finetuned_outputs.csv"))
    save_probs_pkl(ft_probs, os.path.join(PROBS_DIR, "finetuned_probs.pkl"))
    del ft_model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
