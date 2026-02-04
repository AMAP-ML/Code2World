from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

from android_world.agents.vimo import vimo_sum, vimo_reward, vimo_reward_t2
import time

import os

from PIL import Image
import re
import json
from typing import Optional, Any, Dict
import io
import requests
import numpy as np
import concurrent.futures
import ast

# 给定各种信息，action和reason,用简洁的文字返回用户的行为，例如"click the + icon", "scroll up"
SUMMARIZATION_PROMPT_TEMPLATE = (
    'You are a professional UI/UX analyst specializing in identifying the semantic of actions on the mobile UI screenshots.'
    '\n\nInputs:\n\nThe (overall) user goal/request is: {goal}\n'
    'The description for the screenshot before the action: {before_elements}\n'
    'The picked action: {action}\n'
    'Based on the reason: {reason}\n\n'
    'Your task is to analyze these elements and describe the precise user action in plain language and return your answer in plain string (e.g., "click the + icon", "scroll up").'
    'Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean string without any comments.'
)


# 用于prompt的提示词模板，将信息填充进模板。最终返回的是干净的字符串，表示用户操作目标的具体执行步骤
def _summarize_prompt(
    goal: str,
    action: str,
    reason: str,
    before_elements: str,
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    goal: The overall goal.
    action: The action picked for the step.
    reason: The reason why pick the action.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  return SUMMARIZATION_PROMPT_TEMPLATE.format(
      goal=goal,
      action=action,
      reason=reason,
      before_elements=before_elements if before_elements else 'Not available',
  )

# 封装了一个智能体，包含step函数，输入操作信息，返回用户操作目标的具体执行步骤的自然语言描述
class T3A(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  # 输入：用户操作目标，选择执行该action的理由，具体操作，操作前UI元素信息
  # 输出：用户操作目标的具体执行步骤
  def step(self, goal, reason, action, before_element_list) -> base_agent.AgentInteractionResult:
    # 输入文本
    if action["action_type"] == "input_text":
        return "Type in "+ action["text"] + '.'
    # 导航到主页
    if action["action_type"] == "navigate_home":
        return "Press home button to go to home page."
    # 导航到上一页
    if action["action_type"] == "navigate_back":
        return "Press the back button to go to last page."
    # 打开某个app
    if action["action_type"] == "open_app":
        app_name = action['app_name']
        return f"Open the app: {app_name}."
    # 点击发送
    if action["action_type"] == "keyboard_enter":
        return 'Press enter.'
    if action["action_type"] == "wait":
        return 'Wait for the screen to stabilize.'
    if action["action_type"] == "status":
        return action["goal_status"] + '.'
    
    # 对于复杂操作，使用LLM生成描述
    summary_prompt = _summarize_prompt(
        goal,
        action,
        reason,
        before_element_list,
    )

    summary, is_safe, raw_response = self.llm.predict(
        summary_prompt,
    )

    return summary
