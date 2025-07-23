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


#%%
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from tqdm import tqdm

# 📍 4bit 양자화 config (base용)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 📍 경로 설정
fine_tuned_model_path = "./lora_output/mistral_lora_no_quant"
base_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
valid_file_path = "fine_tuned_valid.jsonl"
output_file_path = "predicted_valid_with_reason_short.jsonl"

# 📍 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 📍 base model 먼저 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    local_files_only=True
)

# %%
# 📍 LoRA adapter 적용
fine_tuned_model_path = "./lora_output/mistral_lora_no_quant"
model = PeftModel.from_pretrained(
    base_model,
    fine_tuned_model_path,
    is_trainable=False,
    local_files_only=True  # 추론 용도면 False
)
model.eval()

# 📍 valid 파일 불러오기
with open(valid_file_path, "r", encoding="utf-8") as f:
    valid_entries = [json.loads(line) for line in f]

# 📍 프롬프트 생성 함수
def build_prompt(entry):
    input_ = entry["input"]
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
        f"Write an overall judgement (1–2 sentences) justifying the overall winner. \n"
        f"Do not start with sentences like 'The overall winner is...'\n"
        f"This explanation must reflect the actual aspect(s) that contributed to the win (e.g., functional purpose, technical uniqueness, or strategic value).\n"
        
    )
    return prompt

# 📍 추론 수행
results = []

for entry in tqdm(valid_entries):
    prompt = build_prompt(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_length = inputs["input_ids"].shape[1]  # 프롬프트 토큰 길이

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )

    # 프롬프트 이후 생성된 토큰만 추출
    generated_tokens = outputs[0][input_length:]
    predicted_reason = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


    entry["output"]["overall_reason"] = predicted_reason
    print(f"Generated reason: {predicted_reason}")
    results.append(entry)


# 📍 결과 저장
with open(output_file_path, "w", encoding="utf-8") as f:
    for item in results:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ 총 {len(results)}건의 예측 완료. 결과 파일: {output_file_path}")

# %%

import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
from tqdm import tqdm

# 1.전체 데이터 로딩 및 필요 필드만 남기기
# df_total = load_and_merge_data("old/trial_full.csv", "new/trial_full.csv")
# df_total = df_total[["fp_reason", "tu_reason", "sv_reason"]]

# 📍 경로
output_file_path = "predicted_valid_with_reason_short.jsonl"
embedding_model_name = "AI-Growth-Lab/PatentSBerta"

# 📍 2. valid 파일 불러오기
with open(output_file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 📍 3. valid 파일을 DataFrame으로 정리
rows = []
for item in data:
    row = {
        "fp_reason": item["input"]["fp_reason"],
        "tu_reason": item["input"]["tu_reason"],
        "sv_reason": item["input"]["sv_reason"],
        "overall_reason": item["output"]["overall_reason"]
    }
    rows.append(row)

df_valid = pd.DataFrame(rows)

# 📍 4. dominant aspect 계산
model = SentenceTransformer(embedding_model_name)

fp_vecs = model.encode(df_valid["fp_reason"].fillna("").tolist(), show_progress_bar=True)
tu_vecs = model.encode(df_valid["tu_reason"].fillna("").tolist(), show_progress_bar=True)
sv_vecs = model.encode(df_valid["sv_reason"].fillna("").tolist(), show_progress_bar=True)
overall_vecs = model.encode(df_valid["overall_reason"].fillna("").tolist(), show_progress_bar=True)

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

df_valid["reason_dominant_aspect"] = dominant

# 📍 4. 비율 통계 출력
count = Counter(dominant)
total = sum(count.values())

print("\n📊 dominant aspect 비율:")
for aspect in ["FP", "TU", "SV"]:
    pct = (count[aspect] / total) * 100
    print(f"  - {aspect}: {count[aspect]}건 ({pct:.1f}%)")

# %%
# 📍 5. 원본 JSONL 데이터에 dominant_aspect 추가
for i in range(len(data)):
    data[i]["dominant_aspect"] = df_valid.loc[i, "reason_dominant_aspect"]

# 📍 6. 결과 덮어쓰기 저장
with open(output_file_path, "w", encoding="utf-8") as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"\n✅ dominant_aspect 추가 완료 및 저장: {output_file_path}")
