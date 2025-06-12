import time

# 전역 단계 이름 정의
STEP_MAP = {
    "fetch": 0,
    "summarize": 1,
    "positioning": 2,
    "diff": 3,
    "strategy": 4,
}

PROCESS_STEPS = [
    "Fetching patent texts",
    "Summarizing patent contents",
    "Analyzing technology positioning",
    "Comparing implementation differences",
    "Generating strategic recommendation"
]

# 상태 표시 전용 함수
def show_progress(step_name: str, status_text, progress_bar):
    step_idx = STEP_MAP[step_name]
    total_steps = len(PROCESS_STEPS)
    percent = int((step_idx + 1) / total_steps * 100)

    # 상태 텍스트와 진행률 표시
    status_text.markdown(f"**Step {step_idx+1}/{total_steps}: {PROCESS_STEPS[step_idx]}**")
    progress_bar.progress(percent)
    time.sleep(0.1)