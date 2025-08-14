# %%
# ✅ 예시 프롬프트와 타겟 키워드 설정
prompt = f"You are a helpful assistant. Answer the question based on the context provided.\n\n\, what is the capital of Japan?"
target_words = ["Paris", "Seoul", "Tokyo"]


base_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
from transformers import AutoTokenizer
import torch, torch.nn.functional as F

# ✅ tokenizer: 확률 코드와 동일
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # (확률 코드와 동일 세팅 유지)

from llm_finetune_attention import load_base_model, load_finetuned_model
# model = load_base_model()
model = load_finetuned_model()

# %%
# --- 1. 토크나이저 출력 확인 ---
print("--- 1. 토크나이저 출력 확인 ---")
print(f"입력 프롬프트: \"{prompt}\"")
print(f"토크나이저 출력 (input_ids): {tokenizer(prompt, return_tensors='pt')['input_ids'][0]}")
print("-" * 30)

# ✅ 여러 타겟 키워드의 토큰 ID 리스트 생성
target_token_ids_list = [tokenizer.encode(word, add_special_tokens=False) for word in target_words]
print("--- 타겟 키워드 토큰 정보 ---")
for word, token_ids in zip(target_words, target_token_ids_list):
    print(f"키워드 \"{word}\" -> 토큰 ID: {token_ids}")
print("-" * 30)

# ✅ 답변 생성 (generate)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
gen = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True
)
full_ids = gen.sequences

generated_text = tokenizer.decode(full_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(f"모델이 생성한 답변: \"{generated_text}\"")
print(f"전체 시퀀스 토큰 ID (프롬프트 + 답변): {full_ids[0].tolist()}")
print("-" * 30)

# ✅ 2. 어텐션 맵 계산 및 점수 산출
with torch.no_grad():
    outputs = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids).to(full_ids.device),
        output_attentions=True
    )
    attn_map = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0]

print("--- 2. 어텐션 맵 계산 단계별 확인 ---")
print(f"어텐션 맵 크기: {attn_map.shape}")
print("-" * 30)

# ✅ 각 타겟 키워드에 대한 점수 합산
input_len = inputs['input_ids'].shape[1]
output_len = full_ids.shape[1] - input_len
output_range = list(range(input_len, input_len + output_len))

results = {}
for word, token_ids in zip(target_words, target_token_ids_list):
    token_positions = []
    # 키워드 토큰이 전체 시퀀스에 포함되어 있는지 확인
    for tid in token_ids:
        pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
        token_positions.extend(pos.tolist())
    
    if token_positions:
        # 어텐션 맵에서 답변 토큰 행과 키워드 토큰 열을 선택 후 합산
        attention_scores_for_target = attn_map[output_range][:, token_positions]
        total_score = attention_scores_for_target.mean().item()
        results[word] = total_score
    else:
        results[word] = 0.0 # 키워드가 시퀀스에 없으면 점수는 0

print("--- 각 키워드별 최종 어텐션 점수 ---")
for word, score in results.items():
    print(f"키워드 \"{word}\": {score:.4f}")
print("-" * 30)
# %%
import matplotlib.pyplot as plt
import numpy as np

# --- 3. 레이어별 어텐션 값 변화 시각화 ---
print("--- 3. 레이어별 어텐션 값 변화 시각화 ---")

with torch.no_grad():
    outputs = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids).to(full_ids.device),
        output_attentions=True
    )
    # 텐서의 튜플 형태인 outputs.attentions를 리스트로 변환
    attn_by_layer = list(outputs.attentions) 
    
# 각 레이어별, 키워드별 어텐션 값을 저장할 딕셔너리 초기화
layer_scores = {word: [] for word in target_words}

num_layers = len(attn_by_layer)

for i in range(num_layers):
    # (배치, 헤드, 시퀀스 길이, 시퀀스 길이) -> 헤드 평균 (시퀀스 길이, 시퀀스 길이)
    attn_map_layer = attn_by_layer[i][0].mean(dim=0)
    
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(pos.tolist())
            
        if token_positions:
            # 답변 토큰이 키워드 토큰에 주는 어텐션 값들의 평균
            attention_score = attn_map_layer[output_range][:, token_positions].mean().item()
            layer_scores[word].append(attention_score)
        else:
            layer_scores[word].append(0.0)

# 시각화
plt.figure(figsize=(10, 6))

for word, scores in layer_scores.items():
    plt.plot(range(num_layers), scores, marker='o', linestyle='-', label=word)

plt.title('Attention Score by Layer for Target Words')
plt.xlabel('Layer')
plt.ylabel('Average Attention Score')
plt.xticks(range(num_layers))
plt.legend()
plt.grid(True)
plt.show()

print("-" * 30)

# %%
# --- 4. 특정 레이어의 헤드별 어텐션 값 시각화 ---
print("--- 4. 특정 레이어의 헤드별 어텐션 값 시각화 ---")

with torch.no_grad():
    outputs = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids).to(full_ids.device),
        output_attentions=True
    )
    # (레이어 수, 배치, 헤드 수, 시퀀스 길이, 시퀀스 길이)
    attn_by_layer = outputs.attentions 

# 가장 높은 점수를 보인 18번째 레이어를 선택
target_layer_idx = 18

head_scores = {word: [] for word in target_words}
num_heads = attn_by_layer[target_layer_idx].shape[1]

# 선택한 레이어의 모든 헤드를 순회
for head_idx in range(num_heads):
    # (헤드, 시퀀스 길이, 시퀀스 길이) -> 현재 헤드 선택
    attn_map_head = attn_by_layer[target_layer_idx][0][head_idx]
    
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(pos.tolist())
            
        if token_positions:
            # 답변 토큰이 키워드 토큰에 주는 어텐션 값들의 평균
            attention_score = attn_map_head[output_range][:, token_positions].mean().item()
            head_scores[word].append(attention_score)
        else:
            head_scores[word].append(0.0)

# 시각화
plt.figure(figsize=(12, 7))
x = np.arange(num_heads)
width = 0.25

fig, ax = plt.subplots(figsize=(15, 8))
bar1 = ax.bar(x - width, head_scores['Paris'], width, label='Paris', color='C0')
bar2 = ax.bar(x, head_scores['Seoul'], width, label='Seoul', color='C1')
bar3 = ax.bar(x + width, head_scores['Tokyo'], width, label='Tokyo', color='C2')

ax.set_ylabel('Average Attention Score')
ax.set_xlabel('Head Index')
ax.set_title(f'Attention Score by Head for Layer {target_layer_idx}')
ax.set_xticks(x)
ax.legend()
plt.grid(axis='y')
plt.show()

print("-" * 30)