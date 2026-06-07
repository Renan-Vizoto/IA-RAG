from app.core.prompt_loader import load_prompt


class TestPromptLoader:

    def test_carrega_rag_system(self):
        prompt = load_prompt("rag_system")
        assert "Dutch Energy" in prompt
        assert "CONTEXTO" in prompt
        assert len(prompt) > 100
