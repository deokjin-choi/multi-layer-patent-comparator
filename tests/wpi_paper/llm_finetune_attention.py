# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, torch.nn.functional as F
import pandas as pd, json, gc, os

base_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
lora_path = "./lora_output/mistral_lora_no_quant"
offload_path = "./offload_dir"
os.makedirs(offload_path, exist_ok=True)

# ✅ tokenizer: 확률 코드와 동일
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # (확률 코드와 동일 세팅 유지)

# ✅ 4bit 양자화: 확률 코드와 동일
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"  # ✅ eager 강제
    ).eval()
    model.config.output_attentions = True  # ✅ 어텐션 반환 활성화
    return model

def load_finetuned_model():
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"  # ✅ eager 강제
    )
    base_for_lora.config.output_attentions = True  # ✅ 어텐션 반환 활성화

    model = PeftModel.from_pretrained(
        base_for_lora,
        lora_path,
        is_trainable=False,
        local_files_only=True
    ).eval()
    model.config.output_attentions = True  # ✅ 어텐션 반환 활성화
    return model

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

# ✅ attention 계산: 확률 코드와 동일한 generate 설정 + forward에 attention_mask 명시
def compute_attention_scores(model, prompt, target_words, target_token_ids_list):
    # 확률 코드와 동일: 마스크 없이 generate 가능 (deterministic)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    full_ids = gen.sequences  # [1, input+gen]
    gen_len = full_ids.shape[1] - input_len
    output_range = list(range(input_len, input_len + gen_len))

    # 결과 텍스트 (프롬프트 제외)
    generated_text = tokenizer.decode(full_ids[0][input_len:], skip_special_tokens=True).strip()
    print("Generated:", generated_text[:200])

    # ⚠️ 여기서부터가 'attention' 핵심: forward 재계산 + attention_mask 명시
    with torch.no_grad():
        full_attention_mask = torch.ones_like(full_ids).to(full_ids.device)  # 패딩 없음 → 전부 1
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            output_attentions=True
        )
        # 레이어, 헤드 평균 = 확률 코드와 독립적 / 우리가 보고 싶은 글로벌 집중도
        attn_map = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0]  # (seq_len, seq_len)

    # 키워드별 attention 평균
    result = {}
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(pos.tolist())
        if token_positions:
            score = attn_map[output_range][:, token_positions].mean().item() # 이게 평균이 되어야 함
            result[word] = round(score, 4)
        else:
            result[word] = 0.0
    return result


# %% 어텐션 처리 루프 함수
def process_model(df_keywords, valid_entries, use_lora=False, tag="base"):
    
    if use_lora:
        model = load_finetuned_model()
    else:
        model = load_base_model()

    all_results = []

    def safe_split(val):
        return [w.strip() for w in str(val).split(",") if w.strip()] if pd.notna(val) else []

    for idx, entry in enumerate(valid_entries):
        print(f"Processing row {idx + 1}/{len(valid_entries)}")

        row = df_keywords.iloc[idx]

        # keyword 전체가 NaN이면 skip
        if pd.isna(row.get("keyword")):
            continue

        fp_words = safe_split(row.get("fp_keyword"))
        tu_words = safe_split(row.get("tu_keyword"))
        sv_words = safe_split(row.get("sv_keyword"))
        all_words = fp_words + tu_words + sv_words
        token_ids_list = [tokenizer.encode(w, add_special_tokens=False) for w in all_words]

        # ✅ prompt 생성 (entry는 jsonl에서 가져온 것 그대로 사용)
        prompt = build_prompt_shared(entry)

        # ✅ 어텐션 점수 계산
        scores = compute_attention_scores(model, prompt, all_words, token_ids_list)

        def avg_score(keywords):
            vals = [scores[w] for w in keywords if w in scores]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        result = {
            "id": idx,
            f"{tag}_fp": avg_score(fp_words),
            f"{tag}_tu": avg_score(tu_words),
            f"{tag}_sv": avg_score(sv_words),
        }
        print(f"Results for row {idx + 1}: {result}")
        all_results.append(result)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt

    # keyword 있는 것 가지고 오기
    df_keywords = pd.read_csv("valid_output_base_finetuend_with_keyword.csv")
    df_keywords.columns = df_keywords.columns.str.strip()

    # Load valid entries from JSONL file
    with open("fine_tuned_valid.jsonl", "r", encoding="utf-8") as f:
        valid_entries = [json.loads(line) for line in f]

    # Run
    df_base = process_model(df_keywords, valid_entries, use_lora=False, tag="base")
    df_base.to_csv("attention_base.csv", index=False)

    # Step 2: Finetuned 모델 나중에 처리
    df_finetuned = process_model(df_keywords, valid_entries, use_lora=True, tag="finetuned")
    df_finetuned.to_csv("attention_finetuned.csv", index=False)

    # Step 3: 두 결과 병합 및 저장
    df_merged = pd.merge(df_base, df_finetuned, on="id")
    df_merged.to_csv("attention_comparison_final.csv", index=False)

    # 변화량 계산
    delta_df = df_merged.copy()
    delta_df["delta_fp"] = delta_df["finetuned_fp"] - delta_df["base_fp"]
    delta_df["delta_tu"] = delta_df["finetuned_tu"] - delta_df["base_tu"]
    delta_df["delta_sv"] = delta_df["finetuned_sv"] - delta_df["base_sv"]

    # 히트맵용 데이터
    heatmap_data = delta_df[["delta_fp", "delta_tu", "delta_sv"]]

    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data,  cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Attention Change (Fine-tuned - Base)")
    plt.show()

    heatmap_data.mean()

    # 🔹 FP, TU, SV 각각의 ΔAttention 분포 가져오기
    fp_deltas = delta_df['delta_fp'].dropna().values
    tu_deltas = delta_df['delta_tu'].dropna().values
    sv_deltas = delta_df['delta_sv'].dropna().values

    # 🔹 KDE Plot 시각화
    plt.figure(figsize=(10, 6))
    sns.kdeplot(fp_deltas, fill=True, label="FP", color="skyblue", linewidth=2)
    sns.kdeplot(tu_deltas, fill=True, label="TU", color="lightgreen", linewidth=2)
    sns.kdeplot(sv_deltas, fill=True, label="SV", color="salmon", linewidth=2)

    plt.axvline(0, linestyle='--', color='black', linewidth=1)
    plt.title("Attention Score Change Distribution per Reasoning Dimension (ΔAttention)")
    plt.xlabel("Attention Change (Fine-tuned – Base)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
