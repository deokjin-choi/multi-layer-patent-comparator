# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sentence_transformers import SentenceTransformer
import random

import os
import sys

# app 폴더가 있는 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from app.utils.llm.llm_factory import get_llm_client
from app.utils.llm.retry_utils import safe_invoke


def load_and_merge_data(old_path, new_path):
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    return df_all

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
    """
    SV dominant 편향 케이스를 선별하고, 학습/검증 세트로 분할합니다.

    조건:
    1) match == True (majority winner와 overall winner가 동일)
    2) fp_winner == tu_winner (단, 둘 다 tie 제외)
    3) fp_winner ∈ ['ours', 'competitor']
    4) overall_winner == fp_winner
    5) dominant_reason == 'SV'
    """

    df_filtered = df[
        (df["match"] == True) &
        (df["fp_winner"] == df["tu_winner"]) &
        (df["fp_winner"].isin(["ours", "competitor"])) &
        (df["overall_winner"] == df["fp_winner"]) &
        (df["reason_dominant_aspect"] == "SV")
    ]

    # Dict 리스트로 변환
    data_all = df_filtered.to_dict(orient="records")

    # 랜덤 셔플 및 분할
    random.seed(seed)
    random.shuffle(data_all)
    split = int(len(data_all) * train_ratio)
    train_data = data_all[:split]
    valid_data = data_all[split:]

    return train_data, valid_data

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

# %%
def regenerate_overall_reason(train_data):
    updated_data = []
    llm = get_llm_client()

    for entry in train_data:    
        prompt = make_fp_tu_centered_prompt(entry)
        new_reason = safe_invoke(llm, prompt, parse_func=str)

        # 새 필드로 저장 (기존 overall_reason은 보존)
        entry['overall_reason_fp_tu'] = new_reason
        updated_data.append(entry)

    return updated_data

# %%
df_raw = load_and_merge_data("old/trial_full.csv", "new/trial_full.csv")
df_step1 = add_majority_vote_and_match(df_raw)
df_final = add_dominant_reason(df_step1)
train_data, valid_data = select_sv_bias_cases(df_final)

# %%
train_data = regenerate_overall_reason(train_data)

# %%
finetune_train_set = []
for e in train_data:
    finetune_train_set.append({
        "input": {
            "fp_winner": e["fp_winner"],
            "fp_reason": e["fp_reason"],
            "tu_winner": e["tu_winner"],
            "tu_reason": e["tu_reason"],
            "sv_winner": e["sv_winner"],
            "sv_reason": e["sv_reason"],
            "overall_winner": e["overall_winner"]
        },
        "output": {
            "overall_reason": e["overall_reason_fp_tu"]  # revised_reason 등
        }
    })

finetune_valid_set = []
for e in valid_data:
    finetune_valid_set.append({
        "input": {
            "fp_winner": e["fp_winner"],
            "fp_reason": e["fp_reason"],
            "tu_winner": e["tu_winner"],
            "tu_reason": e["tu_reason"],
            "sv_winner": e["sv_winner"],
            "sv_reason": e["sv_reason"],
            "overall_winner": e["overall_winner"]
        },
        "output": {
            "overall_reason": ""  # 검증 데이터는 출력이 없음
        }
    })

import json

# 🔸 train.jsonl로 저장
with open("fine_tuned_train.jsonl", "w", encoding="utf-8") as f:
    for item in finetune_train_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# 🔹 valid.jsonl로 저장
with open("fine_tuned_valid.jsonl", "w", encoding="utf-8") as f:
    for item in finetune_valid_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# %%
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

# 📍 모델 로컬 경로
BASE_MODEL_PATH = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"

# 📍 데이터 로딩
dataset = load_dataset("json", data_files={
    "train": "fine_tuned_train.jsonl",
    "validation": "fine_tuned_valid.jsonl"
})

# 📍 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 📍 토크나이즈 함수
def tokenize(example):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

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

# 📍 토큰화 수행
tokenized_dataset = dataset.map(tokenize, remove_columns=["input", "output"], batched=True)
print("🔧 토큰화 완료")

# 📍 BitsAndBytes 4bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 📍 Mistral-7B 모델을 4bit로 로드
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# 📍 LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
print("✅ 모델 + LoRA 구성 완료")

# %%

# 📍 데이터로더 구성
from torch.utils.data import DataLoader

# ✅ collate_fn 정의: 리스트 형태의 데이터를 텐서로 변환
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ✅ DataLoader에 collate_fn 적용
train_loader = DataLoader(tokenized_dataset["train"], batch_size=1, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(tokenized_dataset["validation"], batch_size=1, collate_fn=collate_fn)

print("🔧 데이터로더 구성 완료")

# 📍 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=1e-4)

# 📍 학습 루프
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

# 📍 모델 저장
output_dir = "./lora_output/mistral_lora_no_quant"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("💾 모델 및 토크나이저 저장 완료:", output_dir)

# 📍 디바이스 매핑 확인
print("\n📍 디바이스 매핑 결과")
for k, v in model.base_model.model.hf_device_map.items():
    print(f"{k:<60} → {v}")