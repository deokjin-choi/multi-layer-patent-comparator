# app/utils/llm/base_llm_client.py

class BaseLLMClient:
    """
    모든 LLM 클라이언트가 상속해야 하는 기본 인터페이스입니다.
    """
    def invoke(self, prompt: str) -> str:
        """
        입력된 prompt에 대해 응답을 생성합니다.
        각 구현체에서 override 되어야 합니다.
        """
        raise NotImplementedError("invoke() 메서드를 구현해야 합니다.")
