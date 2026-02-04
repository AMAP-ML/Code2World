# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base agent."""

import abc  # 用于定义抽象基类
import dataclasses
import logging
import time
from typing import Any

from android_world.env import interface

# 一个数据类，用于表示代理与环境单次交互的结果：
@dataclasses.dataclass()
class AgentInteractionResult:
  """Result of a single agent interaction with the environment.

  Attributes:
    done: Whether the agent indicates the entire session is done; i.e. this is
      the last interaction with the environment and the session will terminate.
    data: Environment and agent data from interaction.
  """

  done: bool
  data: dict[str, Any]

# 与环境交互的代理的基类，提供了代理与环境交互的基本框架
class EnvironmentInteractingAgent(abc.ABC):
  """Base class for an agent that directly interacts with and acts on the environment.

  This class provides flexibility in agent design, allowing developers to define
  custom action spaces and interaction methods without being confined to a
  specific approach.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = '',
      transition_pause: float | None = 1.0,
  ):
    """Initializes the agent.

    Args:
      env: The environment.
      name: The agent name.
      transition_pause: The pause before grabbing the state. This is required
        because typically the agent is grabbing state immediatley after an
        action and the screen is still changing. If `None` is provided, then it
        uses "auto" mode which dynamically adjusts the wait time based on
        environmental feedback.

    Raises:
      ValueError: If the transition pause is negative.
    """
    self._env = env  # 环境对象，类型为 interface.AsyncEnv
    self._name = name  # 代理名称
    # 动作执行后等待状态稳定的时间（秒），默认为1.0。如果为None，则使用自动模式，根据环境反馈动态调整等待时间
    if transition_pause is not None and transition_pause < 0:
      raise ValueError(
          f'transition_pause must be non-negative, got {transition_pause}'
      )
    self._transition_pause = transition_pause
    # 最大步骤数，初始为None
    self._max_steps = None

  # 提供了对私有属性 _transition_pause 、 _env 和 _name 的访问和修改功能。
  @property
  def transition_pause(self) -> float | None:
    return self._transition_pause

  @transition_pause.setter
  def transition_pause(self, transition_pause: float | None) -> None:
    self._transition_pause = transition_pause

  @property
  def env(self) -> interface.AsyncEnv:
    return self._env

  @env.setter
  def env(self, env: interface.AsyncEnv) -> None:
    self._env = env

  def set_max_steps(self, max_steps: int) -> None:
    self._max_steps = max_steps

  # 重置代理状态，调用env的reset方法，并可选择是否返回主页。
  def reset(self, go_home: bool = False) -> None:
    """Resets the agent."""
    self.env.reset(go_home=go_home)

  # 用于在动作执行后获取稳定的环境状态
  def get_post_transition_state(self) -> interface.State:
    """Convenience function to get the agent state after the transition."""
    # 如果没有暂停时间，则等待环境状态稳定
    if self._transition_pause is None:
      logging.info('Waiting for screen to stabilize before grabbing state...')
      start = time.time()
      state = self.env.get_state(wait_to_stabilize=True)
      logging.info('Fetched after %.1f seconds.', time.time() - start)
      return state
    # 否则，暂停指定时间后获取状态
    else:
      time.sleep(self._transition_pause)
      logging.info(
          'Pausing {:2.1f} seconds before grabbing state.'.format(
              self._transition_pause
          )
      )
      return self.env.get_state(wait_to_stabilize=False)

  # 抽象方法，必须由子类实现。它定义了代理在环境中执行一步的接口
  @abc.abstractmethod
  def step(self, goal: str) -> AgentInteractionResult:
    """Performs a step of the agent on the environment.

    Args:
      goal: The goal.

    Returns:
      Done and agent & observation data.
    """

  @property
  def name(self) -> str:
    return self._name

  @name.setter
  def name(self, name: str) -> None:
    self._name = name
