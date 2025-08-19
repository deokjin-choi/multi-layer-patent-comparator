import os
import sys
import json
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW

# === 경로 세팅 (실행 위치와 무관하게 __file__ 기준으로 설정) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # nlp_paper/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../..")) # patent_compare
PEFT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../peft"))   # peft/

sys.path.append(PROJECT_ROOT)  # app 모듈 불러오기 가능

from app.utils.llm.llm_factory import get_llm_client
from app.utils.llm.retry_utils import safe_invoke


# -------------------------------
# 데이터 준비 관련 함수
# -------------------------------
def load_and_merge_data(old_path, new_path):
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)
    return pd.concat([df_old, df_new], ignore_index=True)

def majority_vote(row):
    votes = [row["fp_winner"], row["tu_winner"], row["sv_winner"]]
    vote_count = Counter(votes)
    top = vote_count.most_common(1)[0][1]
    top_winners = [k for k, v in vote_count.items() if v == top]
    return "tie" if len(top_winners) > 1 else top_winners[0]

def add_majority_vote_and_match(df):
    df["majority_winner"] = df.apply(majority_vote, axis=1)
    df["match"] = df["overall_winner"] == df["majority_winner"]
    return df

def add_dominant_reason(df, model_name="AI-Growth-Lab/PatentSBerta"):
    model = SentenceTransformer(model_name)

    fp_vecs = model.encode(df["fp_reason"].fillna("").tolist(), show_progress_bar=True)
    tu_vecs = model.encode(df["tu_reason"].fillna("").tolist(), show_progress_bar=True)
    sv_vecs = model.encode(df["sv_reason"].fillna("").tolist(), show_progress_bar=True)
    overall_vecs = model.encode(df["overall_reason"].fillna("").tolist(), show_progress_bar=True)

    FP_mean = np.mean(fp_vecs, axis=0)
    TU_mean = np.mean(tu_vecs, axis=0)
    SV_mean = np.mean(sv_vecs, axis=0)

    fp_sim = cosine_similarity(overall_vecs, [FP_mean]).flatten()
    tu_sim = cosine_similarity(overall_vecs, [TU_mean]).flatten()
    sv_sim = cosine_similarity(overall_vecs, [SV_mean]).flatten()

    dominant = []
    for i in range(len(fp_sim)):
        sim_dict = {"FP": fp_sim[i], "TU": tu_sim[i], "SV": sv_sim[i]}
        dominant.append(max(sim_dict, key=sim_dict.get))
    
    df["reason_dominant_aspect"] = dominant
    return df

def select_sv_bias_cases(df, seed=42, train_ratio=0.7):
    """편향 케이스 선별 후 학습/검증 분리"""
    df_filtered = df[
        (df["match"] == True) &
        (df["fp_winner"] == df["tu_winner"]) &
        (df["fp_winner"].isin(["ours", "competitor"])) &
        (df["overall_winner"] == df["fp_winner"]) &
        (df["reason_dominant_aspect"] == "SV")
    ]
    data_all = df_filtered.to_dict(orient="records")

    random.seed(seed)
    random.shuffle(data_all)
    split = int(len(data_all) * train_ratio)
    return data_all[:split], data_all[split:]

def make_fp_tu_centered_prompt(entry):
    prompt = f"""
You are an AI patent evaluation assistant.

Given the following evaluations for Functional Purpose and Technical Uniqueness, explain why the final overall winner is **{entry['overall_winner']}**.

Use clear and concise technical language.
Do NOT mention Strategic Value, scalability, or business impact.

---
[Functional Purpose]
Winner: {entry['fp_winner']}
Our Patent: {entry['fp_ours']}
Competitor Patent: {entry['fp_competitor']}
Reason: {entry['fp_reason']}

[Technical Uniqueness]
Winner: {entry['tu_winner']}
Our Patent: {entry['tu_ours']}
Competitor Patent: {entry['tu_competitor']}
Reason: {entry['tu_reason']}

---
[Overall Reason]
""".strip()
    return prompt


def regenerate_overall_reason(train_data):
    """기존 overall_reason 대신 FP/TU 기반 reason 생성"""
    llm = get_llm_client()
    updated_data = []
    for entry in train_data:    
        prompt = make_fp_tu_centered_prompt(entry)
        new_reason = safe_invoke(llm, prompt, parse_func=str)
        entry['overall_reason_fp_tu'] = new_reason
        updated_data.append(entry)
    return updated_data


# -------------------------------
# 메인 학습 루프
# -------------------------------
def run_finetuning():
    # === 1. 데이터 준비 ===
    df_raw = load_and_merge_data(
        os.path.join(PROJECT_ROOT, "nlp_paper/data/prompts/origin/trial_full.csv"),
        os.path.join(PROJECT_ROOT, "nlp_paper/data/prompts/revised/trial_full.csv")
    )
    df_step1 = add_majority_vote_and_match(df_raw)
    df_final = add_dominant_reason(df_step1)
    train_data, valid_data = select_sv_bias_cases(df_final)
    train_data = regenerate_overall_reason(train_data)

    # === 2. JSONL 저장 ===
    train_path = os.path.join(PEFT_DIR, "fine_tuned_train.jsonl")
    valid_path = os.path.join(PEFT_DIR, "fine_tuned_valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            json.dump({
                "input": {
                    "fp_winner": item["fp_winner"],
                    "fp_reason": item["fp_reason"],
                    "tu_winner": item["tu_winner"],
                    "tu_reason": item["tu_reason"],
                    "sv_winner": item["sv_winner"],
                    "sv_reason": item["sv_reason"],
                    "overall_winner": item["overall_winner"]
                },
                "output": {
                    "overall_reason": item["overall_reason_fp_tu"]
                }
            }, f, ensure_ascii=False)
            f.write("\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data:
            json.dump({
                "input": {
                    "fp_winner": item["fp_winner"],
                    "fp_reason": item["fp_reason"],
                    "tu_winner": item["tu_winner"],
                    "tu_reason": item["tu_reason"],
                    "sv_winner": item["sv_winner"],
                    "sv_reason": item["sv_reason"],
                    "overall_winner": item["overall_winner"]
                },
                "output": {"overall_reason": ""}
            }, f, ensure_ascii=False)
            f.write("\n")

    print("💾 학습/검증 데이터 저장 완료:", train_path, valid_path)

    # === 3. HuggingFace dataset 로드 ===
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": valid_path
    })

    # === 4. Tokenizer / Model 준비 ===
    BASE_MODEL_PATH = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(example):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for i in range(len(example["input"])):
            input_ = example["input"][i]
            output_ = example["output"][i]
            prompt = (
                f"### Input:\n"
                f"fp_winner: {input_['fp_winner']}\n"
                f"fp_reason: {input_['fp_reason']}\n"
                f"tu_winner: {input_['tu_winner']}\n"
                f"tu_reason: {input_['tu_reason']}\n"
                f"sv_winner: {input_['sv_winner']}\n"
                f"sv_reason: {input_['sv_reason']}\n"
                f"overall_winner: {input_['overall_winner']}\n\n"
                f"### Output:\n"
            )
            answer = output_["overall_reason"]
            full_text = prompt + answer

            tokenized = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=768
            )
            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
            labels_list.append(tokenized["input_ids"])
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    tokenized_dataset = dataset.map(tokenize, remove_columns=["input", "output"], batched=True)
    print("🔧 토큰화 완료")

    # === 5. 모델 로딩 (LoRA 적용) ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    print("✅ 모델 + LoRA 구성 완료")

    # === 6. 데이터로더 ===
    def collate_fn(batch):
        input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
        labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_loader = DataLoader(tokenized_dataset["train"], batch_size=1, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(tokenized_dataset["validation"], batch_size=1, collate_fn=collate_fn)
    print("🔧 데이터로더 구성 완료")

    # === 7. 학습 루프 ===
    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3):
        print(f"\n🔁 Epoch {epoch + 1}")
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} 완료 - 평균 Loss: {avg_loss:.4f}")

    # === 8. 모델 저장 ===
    output_dir = os.path.join(PEFT_DIR, "lora_output", "mistral_lora_quant")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("💾 모델 및 토크나이저 저장 완료:", output_dir)

    # === 9. 디바이스 매핑 확인 ===
    print("\n📍 디바이스 매핑 결과")
    for k, v in model.base_model.model.hf_device_map.items():
        print(f"{k:<60} → {v}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    run_finetuning()
