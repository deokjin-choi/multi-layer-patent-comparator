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
        f"Write an overall judgement (1–2 sentences) justifying the overall winner. \n"
        f"Do not start with sentences like 'The overall winner is...'\n"
        f"This explanation must reflect the actual aspect(s) that contributed to the win (e.g., functional purpose, technical uniqueness, or strategic value).\n"
    )

def get_output_attention_scores(model_path, model_name, prompt, target_words, target_token_ids_list, use_lora=False):
    print(f"\n🚀 Loading {model_name} model ...")

    if use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map="auto",
            offload_folder=offload_path,
            low_cpu_mem_usage=True
        ).eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id  # ✅ 패딩 명시적 설정

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    input_len = input_ids.shape[1]

    # ✅ 출력 생성 (generate는 attention 안 줌)
    gen = model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    full_ids = gen.sequences
    gen_len = full_ids.shape[1] - input_len
    output_range = list(range(input_len, input_len + gen_len))

    generated_text = tokenizer.decode(gen.sequences[0][input_len:], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    # ✅ attention 재계산 (forward로)
    with torch.no_grad():
        outputs = model(input_ids=full_ids, output_attentions=True)
        attn_map = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0]  # (seq_len, seq_len)

    # ✅ 단어별 attention 점수 계산
    result = {}
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            positions = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(positions.tolist())
        if token_positions:
            score = attn_map[output_range][:, token_positions].sum().item()
            result[word] = round(score, 4)
        else:
            result[word] = 0.0

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ {model_name} Output Attention:", result)
    return result

# ✅ 토크나이저
tokenizer = AutoTokenizer.from_pretrained(original_model_path)
tokenizer.pad_token = tokenizer.eos_token

# ✅ 키워드 목록
target_words = [
    "distribution", "washing", "tank",
    "movable", "vane",  "mounting", 
    "customer",  "market", "performance"
]
target_token_ids_list = [tokenizer.encode(w, add_special_tokens=False) for w in target_words]

# ✅ 프롬프트
prompt = build_prompt(entry)

# ✅ 실행: 원본 모델
scores_original = get_output_attention_scores(original_model_path, "Original", prompt, target_words, target_token_ids_list)
print(scores_original)

# ✅ 실행: 파인튜닝 모델
scores_finetuned = get_output_attention_scores(fine_tuned_model_path, "Fine-tuned", prompt, target_words, target_token_ids_list, use_lora=True)
print(scores_finetuned)

# ✅ 출력 비교
df = pd.DataFrame([scores_original, scores_finetuned], index=["Original", "Fine-tuned"])
print("\n📊 Output Attention Score Comparison:\n", df)
