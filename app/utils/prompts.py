# app/utils/prompts.py

import yaml
import os

'''
def load_prompt(category: str, version: str = "v1") -> str:
    path = f"prompts/{category}/{version}.yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        return content["template"]
'''

def load_prompt(category: str, version: str = "v1") -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 기준
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))  # app/utils → 루트로 이동
    path = os.path.join(project_root, "prompts", category, f"{version}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        return content["template"]
