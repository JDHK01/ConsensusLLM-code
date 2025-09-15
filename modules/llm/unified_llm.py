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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE, OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from openai import OpenAI
from typing import Dict, Any, Optional


class UnifiedLLM:
    """
    统一的大语言模型接口，支持多种模型提供商。

    支持的提供商：
    - OpenAI (gpt-3.5-turbo, gpt-4, 等)
    - DeepSeek (deepseek-chat, deepseek-reasoner)
    - Kimi (moonshot-v1-8k, moonshot-v1-32k)
    - 通义千问 (qwen-turbo, qwen-plus, qwen-max)
    - 其他兼容OpenAI API的提供商
    """

    # 预设的模型信息
    PROVIDERS = {
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-3.5-turbo'
        },
        'deepseek': {
            'base_url': 'https://api.deepseek.com',
            'default_model': 'deepseek-chat'
        },
        'kimi': {
            'base_url': 'https://api.moonshot.cn/v1',
            'default_model': 'moonshot-v1-8k'
        },
        'tongyi': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'default_model': 'qwen-turbo'
        },
        # 扩展的模版
        'custom': {
            'base_url': None,
            'default_model': None
        }
    }

    def __init__(self,
                 key: str,
                 provider: str = 'deepseek',
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 system_message: Optional[str] = None,
                 temperature: float = 0.7,
                 keep_memory: bool = True):
        """
        初始化统一LLM类。

        Args:
            key (str): API密钥
            provider (str): 提供商名称 ('openai', 'deepseek', 'kimi', 'tongyi', 'custom')
            model (str, optional): 模型名称，如果为None则使用提供商默认模型
            base_url (str, optional): API基础URL，如果为None则使用提供商默认URL
            system_message (str, optional): 系统消息
            temperature (float): 生成温度 (default: 0.7)
            keep_memory (bool): 是否保持对话记忆 (default: True)
        """
        self._key = key
        self._provider = provider
        if provider in self.PROVIDERS:
            provider_config = self.PROVIDERS[provider]
            self._base_url = base_url or provider_config['base_url']
            self._model = model or provider_config['default_model']
        else:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported providers: {list(self.PROVIDERS.keys())}")
        # 初始化客户端
        self._client = OpenAI(
            api_key=key,
            base_url=self._base_url
        )

        self._temperature = temperature
        self._keep_memory = keep_memory

        # 初始化记忆和历史列表
        self._memories = []
        self._history = []

        # 添加系统消息（需要在 _memories 初始化后）
        if system_message:
            self.memories_update(role="system", content=system_message)

    @classmethod
    def create_openai(cls, key: str, model: str = 'gpt-3.5-turbo', **kwargs):
        """便捷方法：创建OpenAI实例"""
        return cls(key=key, provider='openai', model=model, **kwargs)

    @classmethod
    def create_deepseek(cls, key: str, model: str = 'deepseek-chat', **kwargs):
        """便捷方法：创建DeepSeek实例"""
        return cls(key=key, provider='deepseek', model=model, **kwargs)

    @classmethod
    def create_kimi(cls, key: str, model: str = 'moonshot-v1-8k', **kwargs):
        """便捷方法：创建Kimi实例"""
        return cls(key=key, provider='kimi', model=model, **kwargs)

    @classmethod
    def create_tongyi(cls, key: str, model: str = 'qwen-turbo', **kwargs):
        """便捷方法：创建通义千问实例"""
        return cls(key=key, provider='tongyi', model=model, **kwargs)

    @classmethod
    def create_custom(cls, key: str, base_url: str, model: str, **kwargs):
        """便捷方法：创建自定义提供商实例"""
        return cls(key=key, provider='custom', base_url=base_url, model=model, **kwargs)

    @property
    def model(self) -> str:
        """获取当前模型"""
        return self._model

    @property
    def memories(self) -> list:
        """
        获取当前记忆。

        Returns:
            list: 记忆列表
        """
        return self._memories

    @property
    def history(self) -> list:
        """
        获取对话历史。

        Returns:
            list: 对话历史列表
        """
        return self._history

    @property
    def keep_memory(self) -> bool:
        """
        获取是否保持记忆的设置。

        Returns:
            bool: 是否保持记忆
        """
        return self._keep_memory

    def set_keep_memory(self, value: Optional[bool] = None):
        """
        设置是否保持记忆。

        Args:
            value (bool, optional): 输入为布尔值直接设定，如果不是就取反
        """
        if isinstance(value, bool):
            self._keep_memory = value
        else:
            self._keep_memory = not self._keep_memory

    def memories_update(self, role: str, content: str):
        """
        更新记忆以设置角色和内容。

        Args:
            role (str): 角色 (system, user, assistant)
            content (str): 内容

        Raises:
            ValueError: 如果提供了无效的角色或角色添加顺序不正确
        """
        # 合理顺序
        # `system`智能出现在0
        # 必须 `user`和 `assitant` 交替
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Unrecognized role: {role}")

        if role == "system" and len(self._memories) > 0:
            raise ValueError('System role can only be added when memories are empty')

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

    def generate_answer(self, input: str, try_times: int = 0, **kwargs) -> str:
        """
        与LLM模型交互并生成答案。

        Args:
            input (str): 提示或用户输入
            try_times (int): 尝试次数 (default: 0)
            **kwargs: 传递给模型的额外参数

        Returns:
            str: 文本输出结果

        Raises:
            ConnectionError: 生成答案时发生错误
        """
        if not self._keep_memory and len(self._memories) > 0:
            # 只保留系统消息（如果有的话）
            system_messages = [msg for msg in self._memories if msg["role"] == "system"]
            self._memories = system_messages

        if try_times == 0:
            self._memories.append({"role": "user", "content": input})
            self._history.append({"role": "user", "content": input})
        else:
            # 重试时移除上次的assistant回复
            if self._memories and self._memories[-1]["role"] == "assistant":
                self._memories = self._memories[:-1]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._memories,
                temperature=self._temperature,
                **kwargs
            )

            # 提取回复内容
            content = response.choices[0].message.content

            self._memories.append({"role": "assistant", "content": content})
            self._history.append({"role": "assistant", "content": content})

            return content

        except Exception as e:
            raise ConnectionError(f"Error in generate_answer: {e}")

    def reset_conversation(self, keep_system: bool = True):
        """
        重置对话，清除记忆和历史。

        Args:
            keep_system (bool): 是否保留系统消息
        """
        if keep_system and self._memories and self._memories[0]["role"] == "system":
            system_msg = self._memories[0]
            self._memories = [system_msg]
            self._history = [system_msg]
        else:
            self._memories = []
            self._history = []

    def get_info(self) -> Dict[str, Any]:
        """
        获取实例信息。

        Returns:
            dict: 包含提供商、模型、URL等信息的字典
        """
        return {
            'provider': self._provider,
            'model': self._model,
            'base_url': self._base_url,
            'temperature': self._temperature,
            'keep_memory': self._keep_memory,
            'conversation_length': len(self._memories)
        }