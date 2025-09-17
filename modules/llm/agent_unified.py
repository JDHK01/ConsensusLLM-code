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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE, OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import re
from .unified_llm import UnifiedLLM
from ..prompt.summarize import summarizer_role
from ..prompt.form import summarizer_output_form

class Agent(UnifiedLLM):
    """
    继承自`UnifiedLLM`，在这个类中进行了扩展，在基础的回答客户端之外加入了总结客户端

    Args:
        position (float): Agent的当前位置
        other_position (list of float): 其他Agent的位置
        key (str): API密钥
        provider (str): LLM提供商 ('openai', 'deepseek', 'kimi', 'tongyi', 'custom')
        name (str, optional): Agent名称
        model (str, optional): 模型名称，如果为None则使用提供商默认模型
        base_url (str, optional): 自定义API URL（仅当provider='custom'时需要）
        temperature (float): 生成温度 (default: 0.7)
        system_message (str, optional): 系统消息
    """

    def __init__(self,
                 # 用于组建智能体的输入
                 position, other_position,
                 # 构建新模型的参数
                 key: str,
                 provider: str = 'deepseek',
                 model: str = None,
                 base_url: str = None,
                 temperature: float = 0.7,
                 system_message: str = None,
                 # 特性参数
                 name: str = None,
                 ):

        super().__init__(
            key=key,
            provider=provider,
            model=model,
            base_url=base_url,
            system_message=system_message,
            temperature=temperature,
            keep_memory=True
        )

        self._name = name
        self._position = position
        self._other_position = other_position
        self._trajectory = [self.position]

        # 独立的总结器
        self._summarizer = UnifiedLLM(
            key=key,
            provider=provider,
            model=model,
            base_url=base_url,
            keep_memory=False,
            system_message=summarizer_role,
        )
        self._summarize_result = ""
        # 输出格式
        self._summarizer_descriptions = summarizer_output_form

    @classmethod
    def create_deepseek_agent(cls, position, other_position, key: str,
                            name=None, model: str = None, **kwargs):
        return cls(
            position=position,
            other_position=other_position,
            key=key,
            provider='deepseek',
            name=name,
            model=model,
            **kwargs
        )

    @classmethod
    def create_kimi_agent(cls, position, other_position, key: str,
                          name=None, model: str = None, **kwargs):
        return cls(
            position=position,
            other_position=other_position,
            key=key,
            provider='kimi',
            name=name,
            model=model,
            **kwargs
        )

    @classmethod
    def create_tongyi_agent(cls, position, other_position, key: str,
                          name=None, model: str = None, **kwargs):
        return cls(
            position=position,
            other_position=other_position,
            key=key,
            provider='tongyi',
            name=name,
            model=model,
            **kwargs
        )

    @classmethod
    def create_openai_agent(cls, position, other_position, key: str,
                          name=None, model: str = None, **kwargs):
        return cls(
            position=position,
            other_position=other_position,
            key=key,
            provider='openai',
            name=name,
            model=model,
            **kwargs
        )

    @classmethod
    def create_custom_agent(cls, position, other_position, key: str,
                          base_url: str, model: str, name=None, **kwargs):
        return cls(
            position=position,
            other_position=other_position,
            key=key,
            provider='custom',
            base_url=base_url,
            model=model,
            name=name,
            **kwargs
        )

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def other_position(self):
        return self._other_position

    @other_position.setter
    def position(self, value):
        self._other_position = value

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def summarize_result(self):
        return self._summarize_result

    def answer(self, input, idx, round, simulation_ind, try_times=0) -> tuple:
        """
        使用LLM模型生成答案。

        Args:
            input (str): 输入文本或提示
            idx: 索引
            round: 轮次
            simulation_ind: 仿真索引
            try_times (int): 尝试次数

        Returns:
            tuple: 索引和Agent的更新位置
        """
        try:
            answer = self.generate_answer(input=input, try_times=try_times)
            self.position = self.parse_output(answer)
            return idx, self.position
        except Exception as e:
            try_times += 1
            if try_times < 3:
                print(f"Agent {self._name} 生成答案时发生错误: {e}, "
                      f"尝试次数: {try_times + 1}/3")
                return self.answer(input=input, idx=idx,
                                 round=round, simulation_ind=simulation_ind,
                                 try_times=try_times)
            else:
                print(f"经过三次尝试，错误仍未解决，输入为:\n'{input}'\n")
                return idx, self.position  # 返回当前位置

    def summarize(self, agent_answers):
        """
        生成Agent答案的总结。

        Args:
            agent_answers (list): Agent答案列表
        """
        if len(agent_answers) == 0:
            self._summarize_result = ""
        else:
            try:
                input = self._summarizer_descriptions.format(agent_answers)
                self._summarize_result = self._summarizer.generate_answer(
                    input = input,
                )
            except Exception as e:
                print(f"总结生成失败: {e}")
                self._summarize_result = ""

    def parse_output(self, output):
        """
        解析输出用于可视化。

        Args:
            output (str): 模型输出

        Returns:
            float: 解析的位置值
        """
        # 提取数字
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', output)
        if matches:
            x = float(matches[-1])
            self._trajectory.append(x)
            return x
        else:
            raise ValueError(f"输出: \n{output}\n 无法解析")

    def reset_agent(self, new_position=None, keep_system=True):
        """
        重置Agent状态。

        Args:
            new_position (float, optional): 新位置，如果为None则保持当前位置
            keep_system (bool): 是否保留系统消息
        """
        if new_position is not None:
            self._position = new_position
            self._trajectory = [self._position]

        self.reset_conversation(keep_system=keep_system)
        self._summarizer.reset_conversation(keep_system=True)
        self._summarize_result = ""

    def get_agent_info(self):
        """
        获取Agent详细信息。

        Returns:
            dict: Agent信息字典
        """
        info = self.get_info()  # 继承自UnifiedLLM的信息
        info.update({
            'name': self._name,
            'position': self._position,
            'trajectory_length': len(self._trajectory),
            'other_agents_count': len(self._other_position) if self._other_position else 0,
            'summarizer_provider': self._summarizer.provider,
            'summarizer_model': self._summarizer.model
        })
        return info