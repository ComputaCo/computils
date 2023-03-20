class CompletionLLM:
    def complete(self, text, *args, **kwargs) -> str:
        raise NotImplementedError()


class ConversationLM:
    def chat(self, messages, *args, **kwargs) -> str:
        raise NotImplementedError()
