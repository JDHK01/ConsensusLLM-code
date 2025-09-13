"""
MIT License

Copyright (c) [2023] [Intelligent Unmanned Systems Laboratory at 
Westlake University]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE, OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from openai import OpenAI

class DeepSeek:
    """
    Initialize the DeepSeek class for interacting with DeepSeek's models.
    DeepSeek provides basic methods for interacting with the model and parsing its
    output.
    """

    def __init__(self, key: str, model: str = 'deepseek-chat',system_message: str = None,
                 temperature: float = 0.7, keep_memory: bool = True, ):
        """
        Initialize the DeepSeek class.

        Args:
            key (str): DeepSeek API key.
            model (str): The model to use (default: deepseek-chat).
            temperature (float): Temperature for text generation (default: 0.7).
            keep_memory (bool): Whether to retain memories (default: True).
        """
        # 模型的参数信息:
        self._key = key
        self._model = model
        self._temperature = temperature
        self._memories = [] # 让模型看的记忆

        self._keep_memory = keep_memory # 控制是否维持记忆
        self._history = [] # 让人看的记忆
        self._cost = 0

        # 新版客户端
        self._client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com"
        )

        # 加入系统信息
        if system_message:
            self.memories_update(role="system", content=system_message)

    def get_memories(self):
        """
        Get the current memories.

        Returns:
            list: List of memories.
        """
        return self._memories

    def get_history(self):
        """
        Get the conversation history.

        Returns:
            list: List of conversation history.
        """
        return self._history

    def get_keep_memory(self):
        """
        得到模型参数: 是否保存历史消息

        Returns:
            bool: 是否保存消息的布尔值
        """
        return self._keep_memory

    def set_keep_memory(self, arg = None):
        """
        设置模型参数: 是否保持布尔值

        Args:
            arg(any): 任意值。如果是布尔值会将`self_keep_memory`取反,
                    否则区反
        """
        if isinstance(arg, bool):
            self._keep_memory = arg
        else:
            self._keep_memory = not self._keep_memory

    def memories_update(self, role: str, content: str):
        """
        Update memories to set roles (system, user, assistant) and content,
        forming a complete memory.
        `self.memories`的格式
        eg. [{'role': 'system', 'content': '你是一个复读机'},\n
             {'role': 'user', 'content': '请你记住“制图之体有六，缺一不可言精”'},\n
             {'role': 'assistant', 'content': '制图六体，缺一不可。'},\n
             {'role': 'user', 'content': '制图之体有六下一句是什么, 回答不超过10个字'},\n
             {'role': 'assistant', 'content': '文成规矩，随变而立功'}]

        Args:
            role (str): Role (system, user, assistant).
            content (str): Content.

        Raises:
            ValueError: If an unrecognized role is provided or if roles are
            added in an incorrect sequence.
        """
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Unrecognized role: {role}")

        if role == "system" and len(self._memories) > 0:
            raise ValueError('System role can only be added when memories are '
                             'empty')
        if (role == "user" and len(self._memories) > 0 and
            self._memories[-1]["role"] == "user"):
            raise ValueError('User role can only be added if the previous '
                             'round was a system or assistant role')
        if (role == "assistant" and len(self._memories) > 0 and
            self._memories[-1]["role"] != "user"):
            raise ValueError('Assistant role can only be added if the previous '
                             'round was a user role')
        self._memories.append({"role": role, "content": content})
        self._history.append({"role": role, "content": content})

    def generate_answer(self, input: str, try_times=0, **kwargs) -> str:
        """
        Interact with the DeepSeek model and generate an answer.

        Args:
            input (str): Prompt or user input.
            try_times (int): Number of attempts (default is 0).
            kwargs: Additional parameters for the model.

        Returns:
            str: Text-based output result.

        Raises:
            ConnectionError: If there's an error in generating the answer.
        """
        if not self._keep_memory:
            self._memories = [self._memories[0]]

        if try_times == 0:
            self._memories.append({"role": "user", "content": input})
            self._history.append({"role": "user", "content": input})
        else:
            if self._memories[-1]["role"] == "assistant":
                self._memories = self._memories[:-1]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._memories,
                temperature=self._temperature,
                **kwargs
            )
            # 取出回复的内容
            self._cost += response.usage.total_tokens
            content = response.choices[0].message.content

            self._memories.append({"role": "assistant", "content": content})
            self._history.append({"role": "assistant", "content": content})

            return content
        except Exception as e:
            raise ConnectionError(f"Error in generate_answer: {e}")