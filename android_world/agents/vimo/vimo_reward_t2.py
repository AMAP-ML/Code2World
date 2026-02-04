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

"""T3A: Text-only Autonomous Agent for Android."""

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
# 给定两个action，找出最好的那个action

PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message), like'
    ' user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by performing'
    ' actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step. At'
    ' each step, a list of descriptions for most UI elements on the'
    ' current screen will be given to you (each element can be specified by an'
    ' index), together with a history of what you have done in previous steps.'
    ' Based on these pieces of information and the goal, you must choose to'
    ' perform one of the action in the following list (action description'
    ' followed by the JSON format) by outputing the action in the correct JSON'
    ' format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    '- If you think the task is not'
    " feasible (including cases like you don't have enough information or can"
    ' not perform some necessary actions), finish by using the `status` action'
    ' with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into an editable text field (specified by its index), this'
    ' action contains clicking the text field, typing in the text and pressing'
    ' the enter, so no need to click on the target field to start:'
    ' `{{"action_type": "input_text", "text": <text_input>, "index":'
    ' <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not installed):'
    ' `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history), try to"
    ' switch to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations\n'
    '- Normally to select some text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

# 给定需要的信息，要求agent从两个action中选择出最好的那个action并给出理由
REWARD_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\n\nThe overall user goal/request is: {goal}'
    '\n\nHere is a history of what you have done so far:\n{history}'
    '\n\nThe current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
    "\nHere are two candidate actions:\n"
    "Action 1: {action_0}, described semantically as {sum_0}. The rationale for this action is: {act_re_0}\n"
    "Action 2: {action_1}, described semantically as {sum_1}. The rationale for this action is: {act_re_1}\n"
    + GUIDANCE
    + '\nHints for making your decision:\n'
    '- Both "more options" buttons and scrolling actions may reveal new content. Evaluate which is more suitable for the goal.\n'
    '- Consider the history of previous actions. If prior steps involved repeated "scroll down" actions, it is more likely that "scroll down" is the correct next step.\n'
    '- If the user goal involves viewing reviews or similar tasks and the current screen already displays such content, "scroll down" may reveal more information.\n'
    + 'Your task is to choose the best action from the two provided.\n'
    '\nNow, provide your judgment in JSON format with the following structure:\n'
    "Reason: A detailed explanation of your choice, considering the hints above.\n"
    "Choice: Action 1 or Action 2.\n"
    # '**You must follow this structure exactly without any comment or code block:**\n'
    'Output using [ ] to enclose the content, which must be in JSON format with the keys Reason and Choice. No other comments are allowed. In other words, you must strictly follow the format below for output:\n'
    '[{{"Reason": "...", "Choice": "Action 1 or Action 2"}}] \n'
    'Your Answer:\n'
)

def _reward_prompt(
    history: list[str],
    goal: str,
    before_elements: str,
    action_0,
    act_re_0,
    sum_0,
    action_1,
    act_re_1,
    sum_1
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    action: Action picked.
    reason: The reason to pick the action.
    goal: The overall goal.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'
  return REWARD_PROMPT_TEMPLATE.format(
      goal=goal,
      history=history,
      ui_elements=before_elements,
      action_0=action_0,
      act_re_0=act_re_0,
      sum_0=sum_0,
      action_1=action_1,
      act_re_1=act_re_1,
      sum_1=sum_1
  )

def extratc_json(output):
    output_u = {}

    # 截取出最外层的[]包裹的内容
    start = output.find('[')
    end = output.rfind(']')
    if start == -1 or end == -1 or end < start:
        return None
    output = output[start:end+1]  

    # Try loading directly
    try:
        output = json.loads(output)
        return output[0]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from triple backticks with optional language
    match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*])\s*```", output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to fallback to JSON-like string in case no code block is matched
        json_str = output.strip()

    # Attempt to parse JSON
    try:
        parsed = json.loads(json_str)
        return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else None
    except json.JSONDecodeError as e:
        print("t2输出解析时出现报错json.JSONDecodeError")
        print(json_str)
        print(e)
        return None
      
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

  def step(self, goal, his, raw_screenshot, image_array_before, before_element_list, action_0, act_re_0, sum_0, action_1, act_re_1, sum_1) -> base_agent.AgentInteractionResult:
    action_prompt = _reward_prompt(
        [
            'Step ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(his)
        ],
        goal,
        before_element_list,
        action_0,
        act_re_0,
        sum_0,
        action_1,
        act_re_1,
        sum_1,
    )
    
    action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
              raw_screenshot,
              image_array_before,
            ]
        )
    
    print("t2的action_output为:", action_output)
    action_output_json = extratc_json(action_output)
    if action_output_json is None:
       return None, None
    reason, judge = action_output_json['Reason'], action_output_json['Choice']
    judge = judge.lower()
    
    return judge, action_output_json

