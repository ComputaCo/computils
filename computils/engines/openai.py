import os
import json


import openai

from computaco_utils.fns.exponential_backoff import exponential_backoff
from computaco_utils.engines.base import CompletionEngine, ConversationEngine


openai.api_key = os.environ["OPENAI_API_KEY"]


class OpenAIEngine:
    params: dict

    def __init__(self, model_name, no_multithreading=False, **api_default_kwargs):
        super().__init__(no_multithreading=no_multithreading)
        self.params = self.PARAM_DEFAULTS[model_name].copy()
        self.params["model"] = model_name
        self.api_default_kwargs = api_default_kwargs

    def token_estimate(messages):
        return int(0.25 * sum(len(message["content"]) for message in messages))

    def prepare_kwargs(self, messages, **kwargs):
        _kwargs = self.params.copy()
        _kwargs.update(
            dict(
                max_tokens=(
                    ChatGPT.DEFAULT_MODEL_MAX_TOKENS
                    - self.token_estimate(messages)
                    - 50
                ),
            )
        )
        _kwargs.update(self.api_default_kwargs)
        _kwargs.update(kwargs)


class TextGPT(OpenAIEngine, CompletionEngine):

    PARAM_DEFAULTS = {
        "text-davinci-003": {"max_tokens": 4097},
        "text-davinci-002": {"max_tokens": 4097},
        "text-davinci-001": {"max_tokens": 2048},
        "text-curie-001": {"max_tokens": 2049},
        "text-babbage-001": {"max_tokens": 2049},
        "text-ada-001": {"max_tokens": 2049},
        "davinci-instruct-beta": {"max_tokens": 2049},
        "code-davinci-002": {"max_tokens": 8001},
        "code-cushman-001": {"max_tokens": 2048},
    }

    def _complete(self, text, *args, **kwargs) -> str:
        kwargs = self.prepare_kwargs(text, **kwargs)
        kwargs["prompt"] = text
        output = exponential_backoff(
            fn=(lambda: openai.Completion.create(kwargs)["choices"][0]["text"]),
            retry_exception=openai.error.RateLimitError,
        )
        return output


class ChatGPT(OpenAIEngine, ConversationEngine):

    PARAM_DEFAULTS = {
        "gpt-3.5-turbo": {"max_tokens": 4096},
        "gpt-4": {"max_tokens": 8192},
    }

    def _chat(self, messages, *args, **kwargs) -> str:
        kwargs = self.prepare_kwargs(messages, **kwargs)
        kwargs["messages"] = messages
        output = exponential_backoff(
            fn=(lambda: openai.Completion.create(kwargs)["choices"][0]["text"]),
            retry_exception=openai.error.RateLimitError,
        )
        return output

    def load_json_export(self, jsonpath):
        # This method is used along with the JSON export script:
        # https://raw.githubusercontent.com/ryanschiang/chatgpt-export/main/exportJSON.min.js

        print(f"Loading conversation from {jsonpath}...")

        with open(jsonpath, "r") as f:
            input_json = json.load(f)

        def traverse_message(message):
            if isinstance(message, list):
                return " ".join(traverse_message(item) for item in message)
            elif isinstance(message, dict):
                return traverse_message(message.get("data", ""))
            else:
                return message

        for chat in input_json["chats"]:
            role = (
                "user" if "type" in chat and chat["type"] == "prompt" else "assistant"
            )
            self.messages.append(
                {"role": role, "content": traverse_message(chat["message"])}
            )
