import json
import logging

from computaco_utils.engines.base import ConversationLM



class Conversation:

    # Message's are shaped: {role: str, content: str}
    messages: list[dict[str, str]]
    chat_lm: ConversationLM
    path: str

    def __init__(self, chat_lm=None, system_message=None, path=None):
        self.chat_lm = chat_lm
        self.path = path

        self.messages = []
        self.load()
        if system_message is not None and len(self.messages) == 0:
            self.messages.append({"role": "system", "content": system_message})
            self.save()

    def chat(self, message: str, *args, chat_lm=None, **kwargs):
        chat_lm = chat_lm or self.chat_lm
        if chat_lm is None:
            return
        self.messages.append({"role": "user", "content": message})
        response = self.chat_lm.chat(self.messages, *args, **kwargs)
        self.messages.append(response)
        self.save()
        return response["content"]

    def load(self, path+None):
        path = path or self.path
        if path is None:
            return
        with open(path, "r") as f:
            self.messages = json.load(f)

    def save(self, path=None):
        path = path or self.path
        if path is None:
            return
        with open(path, "w") as f:
            json.dump(self.messages, f)
