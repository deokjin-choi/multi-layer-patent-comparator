# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM 
import gc
import os

# ✅ 모델 경로
original_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
fine_tuned_model_path = "./lora_output/mistral_lora_no_quant"

# 오프로드 경로 지정 (필요 시 폴더 직접 생성)
offload_path = "./offload_dir"
os.makedirs(offload_path, exist_ok=True)

# ✅ 입력 예시
entry = {
    "fp_winner": "ours", 
    "fp_reason": "The first patent (US10231596B2) optimizes washing water distribution in the entire washing tank area, while the second patent (US11219349B2) only stabilizes rotation of wash arm assembly during dynamic operational conditions.", 
    "tu_winner": "ours", 
    "tu_reason": "The first patent (US10231596B2) utilizes a unique movable vane to distribute washing water, whereas the second patent (US11219349B2) uses a radial supporting portion in its mounting unit.", 
    "sv_winner": "ours", 
    "sv_reason": "The first patent (US10231596B2) improves cleaning effectiveness, potentially leading to customer satisfaction and market differentiation, while the second patent (US11219349B2) enhances product performance for consistent cleaning results.", 
    "overall_winner": "ours"
}

# ✅ 프롬프트 생성 함수
def build_prompt(entry):
    return (
        f"### Input:\n"
        f"fp_winner: {entry['fp_winner']}\n"
        f"fp_reason: {entry['fp_reason']}\n"
        f"tu_winner: {entry['tu_winner']}\n"
        f"tu_reason: {entry['tu_reason']}\n"
        f"sv_winner: {entry['sv_winner']}\n"
        f"sv_reason: {entry['sv_reason']}\n"
        f"overall_winner: {entry['overall_winner']}\n\n"
        f"### Output:\n"
    )

# ✅ Attention 계산 함수
def get_attention_scores(model_path, model_name, input_ids, target_words, target_token_ids, use_lora=False):
    print(f"\n🚀 Loading {model_name} model ...")

    if use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,                     # LoRA adapter 경로 (여기에 base도 포함)
            torch_dtype=torch.float16,      # 또는 float32 가능
            attn_implementation="eager",    # 필요시 유지
            output_attentions=True,
            device_map="auto",
            #offload_folder=offload_path,    # offload_dir 아님! 주의
            low_cpu_mem_usage=False
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.float16,             # 또는 float32
            attn_implementation="eager",
            output_attentions=True,
            device_map="auto",
            offload_folder=offload_path,
            low_cpu_mem_usage=True
        ).eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
        attentions = outputs.attentions

    if attentions is None:
        print(f"❌ {model_name} returned no attention.")
        return {w: 0.0 for w in target_words}

    attn_map = torch.stack(attentions).mean(dim=0).mean(dim=1)[0]  # (seq_len, seq_len)

    result = {}
    for word, tid in zip(target_words, target_token_ids):
        try:
            token_positions = (input_ids[0] == tid).nonzero(as_tuple=True)[0]
            score = attn_map[:, token_positions].sum().item()
            result[word] = round(score, 4)
        except:
            result[word] = 0.0

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ {model_name} Attention:", result)
    return result

# ✅ Tokenizer 로딩
tokenizer = AutoTokenizer.from_pretrained(original_model_path)
tokenizer.pad_token = tokenizer.eos_token

# ✅ 입력 토큰화
prompt = build_prompt(entry)
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

# ✅ 관심 단어 설정
target_words = [
    "distribution", "washing", "tank",
    "movable", "vane", "radial", "supporting", "mounting", "unit",
    "customer", "satisfaction", "market", "differentiation", "performance", "effectiveness"
]
target_token_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in target_words]

# ✅ 원본 모델 실행
scores_original = get_attention_scores(original_model_path, "Original", input_ids, target_words, target_token_ids)

# ✅ Fine-tuned 모델 실행 (LoRA 병합 후)
# scores_finetuned = get_attention_scores(fine_tuned_model_path, "Fine-tuned", input_ids, target_words, target_token_ids, use_lora=True)

# %%

# ✅ 비교 결과 출력
df = pd.DataFrame([scores_original, scores_finetuned], index=["Original", "Fine-tuned"])
print("\n📊 Attention Score Comparison:\n", df)
