# This script calculates the mean token length of a fine-tuned dataset for Mistral-7B-v0.1.

from transformers import AutoTokenizer
import json

# ✅ Mistral 토크나이저 로딩
BASE_MODEL_PATH = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)

# ✅ jsonl 파일 읽기
file_path = "fine_tuned_train.jsonl"
token_lengths = []

with open(file_path, "r") as f:
    for line in f:
        item = json.loads(line)
        input_ = item["input"]
        output_ = item["output"]

        # Prompt 구성
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
        full_text = prompt + output_["overall_reason"]

        # 토큰 길이 측정
        tokenized = tokenizer(full_text, truncation=False)
        token_len = len(tokenized["input_ids"])
        token_lengths.append(token_len)

# ✅ 간단한 통계 출력
import numpy as np
print(f"샘플 수: {len(token_lengths)}")
print(f"평균 토큰 길이: {np.mean(token_lengths):.1f}")
print(f"최대: {np.max(token_lengths)} / 최소: {np.min(token_lengths)}")
