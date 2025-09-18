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

import re
import numpy as np
from .unified_llm import UnifiedLLM
from ..prompt.summarize import summarizer_role
from ..prompt.form import summarizer_output_form

class Agent2D(UnifiedLLM):
    """
    A class representing a 2D agent with position control.

    Args:
        position (tuple): Current position of the agent (x, y).
        other_position (list of tuples): Positions of other agents.
        key (str): API key for the LLM model.
        provider (str): LLM provider ('openai', 'deepseek', 'kimi', 'tongyi', 'custom').
        name (str): Name of the agent (optional).
        model (str): Model name (default uses provider's default model).
        base_url (str, optional): Custom API URL (only when provider='custom').
        temperature (float):
            Temperature for text generation (default is 0.7).
        keep_memory (bool):
            Whether to keep a memory of conversations (default is False).
    """
    
    def __init__(self, position, other_position, key: str,
                 provider: str = 'deepseek', name=None,
                 model: str = None, base_url: str = None,
                 temperature: float = 0.7, keep_memory=False):
        super().__init__(key=key, provider=provider, model=model,
                         base_url=base_url, temperature=temperature,
                         keep_memory=keep_memory)
        self._name = name
        self._velocity = np.zeros(2)  # Current velocity of the agent
        self._max_traction_force = 50  # Maximum traction force of the agent (N)
        self._max_velocity = 3  # Maximum velocity of the agent (m/s)
        self._m = 15  # Mass of the agent (kg)
        self._mu = 0.02  # Friction coefficient
        # PID Parameters
        self.Kp = np.array([1.2, 1.2], dtype=np.float64)
        self.Ki = np.array([0.0, 0.0], dtype=np.float64)
        self.Kd = np.array([6, 6], dtype=np.float64)
        self.prev_error = np.array([0, 0], dtype=np.float64)
        self.integral = np.array([0, 0], dtype=np.float64)
        self._target_position = None  # Target position of the agent
        self._position = position  # Current position of the agent
        self._other_position = other_position  # Positions of other agents
        self._trajectory = []  # Record the agent's movement trajectory
        self._target_trajectory = []  # Record the agent's target trajectory
        self._summarizer = UnifiedLLM(key=key, provider=provider,
                                      model=model, base_url=base_url,
                                      keep_memory=False,
                                      system_message=summarizer_role)
        self._summarize_result = ""
        self._summarizer_descriptions = summarizer_output_form

    @classmethod
    def create_deepseek_agent(cls, position, other_position, key: str,
                              name=None, model: str = None, **kwargs):
        """Create a DeepSeek agent."""
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
        """Create a Kimi agent."""
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
        """Create a Tongyi agent."""
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
        """Create an OpenAI agent."""
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
        """Create a custom provider agent."""
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

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def target_trajectory(self):
        return self._target_trajectory

    @property
    def target_position(self):
        return self._target_position

    @other_position.setter
    def other_position(self, value):
        self._other_position = value

    @property
    def summarize_result(self):
        return self._summarize_result

    def answer(self, input, idx, round, simulation_ind, try_times=0) -> tuple:
        """
        Generate an answer using the DeepSeek model.

        Args:
            input (str): Input text or prompt.
            idx: Index.
            round: Round.
            simulation_ind: Simulation index.
            try_times (int): Number of times the answer generation is attempted.

        Returns:
            tuple: Index and the target position (x, y).
        """
        try:
            answer = self.generate_answer(input=input, try_times=try_times)
            self._target_position = self.parse_output(answer)
            self._target_trajectory.append(self._target_position)
            return idx, self._target_position
        except Exception as e:
            try_times += 1
            if try_times < 3:
                print(f"An error occurred when agent {self._name} tried to "
                      f"generate answers: {e},try_times: {try_times + 1}/3.")
                return self.answer(input=input, idx=idx, round=round, 
                                   simulation_ind=simulation_ind, 
                                   try_times=try_times)
            else:
                print("After three attempts, the error still remains "
                      f"unresolved, the input is:\n'{input}'\n.")
                return idx, self._target_position

    def summarize(self, agent_answers):
        """
        Generate a summary of agent answers.

        Args:
            agent_answers (list): List of agent answers.
        """
        if len(agent_answers) == 0:
            self._summarize_result = ""
        else:
            self._summarize_result = self._summarizer.generate_answer(
                self._summarizer_descriptions.format(agent_answers))

    def parse_output(self, output):
        """
        Parse the output for visualization.

        Args:
            output (str): Model's output.

        Returns:
            tuple: Parsed position value (x, y).
        """
        # First try to find "Position:" section
        position_match = re.search(r'[Pp]osition\s*:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
        if position_match:
            position_text = position_match.group(1).strip()
            # Try to extract coordinates from position text
            numbers = re.findall(r'[-+]?\d*\.?\d+', position_text)
            if len(numbers) >= 2:
                x = float(numbers[0])
                y = float(numbers[1])
                return (x, y)

        # Fallback: try to find coordinates in parentheses
        matches = re.findall(r'\((.*?)\)', output)
        if matches:
            for match in reversed(matches):  # Try from last to first
                numbers = re.findall(r'[-+]?\d*\.?\d+', match)
                if len(numbers) >= 2:
                    x = float(numbers[0])
                    y = float(numbers[1])
                    return (x, y)

        # Fallback: try to find any two consecutive numbers
        all_numbers = re.findall(r'[-+]?\d*\.?\d+', output)
        if len(all_numbers) >= 2:
            # Take the last two numbers
            x = float(all_numbers[-2])
            y = float(all_numbers[-1])
            return (x, y)

        # If all fails, return current position as fallback
        print(f"Warning: Could not parse position from output, using current position: {self._position}")
        print(f"Output was: {output}")
        return self._position if self._position else (0.0, 0.0)

    def move(self, time_duration: float):
        """
        Move the agent based on PID control.

        Args:
            time_duration (float): Time duration for the movement.
        """
        if self._target_position is None:
            print("Target not set!")
            return
        error = np.array(self._target_position) - np.array(self._position)
        self.integral += error * time_duration
        derivative = (error - self.prev_error) / time_duration
        force = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self._max_traction_force:
            force = (force / force_magnitude) * self._max_traction_force
        # friction_force = -self._mu * self._m * 9.8 * np.sign(self._velocity) if abs(
        #   np.linalg.norm(self._velocity)) > 0.1 else 0
        friction_force = 0
        net_force = force + friction_force
        acceleration = net_force / self._m
        self._velocity += acceleration * time_duration
        # Limit the velocity
        velocity_magnitude = np.linalg.norm(self._velocity)
        if velocity_magnitude > self._max_velocity:
            self._velocity = (self._velocity / velocity_magnitude) * self._max_velocity
        self._position += self._velocity * time_duration + 0.5 * acceleration * time_duration ** 2
        self._position = tuple(np.round(self._position, 2))
        self.prev_error = error
        self._trajectory.append(self._position)
        # print(f"{self._name} position: {self._position}, "
        #       f"target: {self._target_position}, velocity: {self._velocity}, "
        #       f"force: {force}, friction_force: {friction_force}, "
        #       f"net_force: {net_force}, acceleration: {acceleration}")
        return self._position

    def get_history(self):
        """
        Get agent's conversation history.

        Returns:
            list: Conversation history list
        """
        return self.history

    def get_agent_info(self):
        """
        Get agent detailed information.

        Returns:
            dict: Agent information dictionary
        """
        info = self.get_info()  # Inherited from UnifiedLLM
        info.update({
            'name': self._name,
            'position': self._position,
            'trajectory_length': len(self._trajectory),
            'other_agents_count': len(self._other_position) if self._other_position else 0,
            'target_position': self._target_position,
            'summarizer_provider': self._summarizer.provider,
            'summarizer_model': self._summarizer.model
        })
        return info