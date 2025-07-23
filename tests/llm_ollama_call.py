import ollama

class LocalOllamaLLMClient:
    """
    Ollama 0.6.8 이후 완전 호환형 LLM Client (mistral / llama3 자동지원)
    """

    def __init__(self, model: str = "mistral"):
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            
        )
        return response['message']['content'].strip()

# 테스트 예제
if __name__ == "__main__":
    # llama3 사용시
    client = LocalOllamaLLMClient(model="mistral")

    # mistral 사용시
    # client = LocalOllamaLLMClient(model="mistral")

    result = client.invoke("Tell me briefly about carbon capture technology.")
    print(result)
