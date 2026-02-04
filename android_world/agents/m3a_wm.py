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

"""A Multimodal Autonomous Agent for Android (M3A)."""
import textwrap
import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

from android_world.agents.vimo import vimo_sum, vimo_reward, vimo_reward_t2
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
from android_world.agents.wm_utils import _post_figure_action, MLLMInferencer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def post_figure_action(image_array, action, html_save_path, idx):
    # inference = MLLMInferencer("gpt-5")
    inference = MLLMInferencer("internvl3-78b")
    returned_array = _post_figure_action(inferencer=inference, image_array=image_array, action=action, html_save_path=html_save_path, idx=idx)
    
    return returned_array

# 定义了 MLLM 的基本角色和任务，是操作 Android 手机的代理，有两种主要任务：回答用户问题、完成手机操作任务
# 定义了可执行的动作列表及 JSON 格式说明
PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish by using the `status` action with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:" 
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

# 提供智能体执行任务时的详细guidance，包括：
# - 通用规则：选择最简单的方法、重试失败操作、切换方案
# - 信息收集：如何导航手机获取信息
# - 动作相关：正确使用各种动作类型
# - 文本操作：选择、删除、复制、粘贴文本的方法
GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
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
    'Action Related:\n'
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
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
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

# 生成动作选择阶段的完整提示。使用 {} 占位符以便动态填充：
# - {goal} : 用户当前目标
# - {history} : 已执行步骤的历史记录
# - {ui_elements} : 当前屏幕的 UI 元素列表
# - {additional_guidelines} : 任务特定的额外指导

ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
    'Here is a history of what you have done so far:\n{history}\n\n'
    'The current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + textwrap.dedent("""
      The output **must** be a single, valid JSON value (no extra text, no explanation, no markdown/code fences). The JSON must be parseable by standard JSON parsers (e.g. `json.loads` in Python).

      Requirements:
      1. Output exactly 3 actions with keys "1", "2", "3" (strings). Example structure:
      {{
        "1": {{"Reason": "...", "Confidence": "...", "Action": {{"action_type": "...", ...}}}},
        "2": {{"Reason": "...", "Confidence": "...", "Action": {{"action_type": "...", ...}}}},
        "3": {{"Reason": "...", "Confidence": "...", "Action": {{"action_type": "...", ...}}}}
      }}

      2. Use **double quotes** for all JSON strings and keys (standard JSON). Do NOT use single quotes for JSON.

      3. All string values (including Reason and any Action fields such as text) must be valid JSON strings. If a string contains double quotes, newline characters, backslashes, or other characters that would break JSON, you MUST escape them using standard JSON escaping (for example: a quote becomes `\"`, a backslash becomes `\\`, a newline becomes `\n`).

      4. Do NOT output trailing commas. Do NOT include comments or non-JSON tokens.

      5. Each "Reason" must be a plain string explaining why that action is valid/plausible.  
        Each "Action" must be a JSON object with at least the field "action_type" (string). If the action includes a "text" field, it must be a JSON string (properly escaped).

      6. Output must be compact or pretty-printed JSON — both are acceptable as long as it is valid JSON.

      7. Add a **Confidence** field for each action:
        - Field name: "Confidence"
        - Type: a numeric value (floating point) between 0.0 and 1.0 (inclusive).
        - Semantics: a confidence score reflecting how likely the action is correct.
        - Use this scale to choose the numeric value:
          - 1.0: Absolute certainty based on clear evidence or explicit rules
          - 0.8–0.9: High confidence with strong supporting evidence
          - 0.6–0.7: Moderate confidence with some ambiguity
          - 0.4–0.5: Low confidence due to significant uncertainty
          - 0.1–0.3: Very low confidence with minimal supporting evidence

      IMPORTANT: Output nothing but the single JSON object. No surrounding explanation, no code fences, no extra newlines before/after.

      Now, based on the earlier extracted candidate actions, output 3 different potential actions following the schema above.
      """)

)


# 生成步骤总结阶段的完整提示。用于让 LLM 总结已执行动作的结果，以便后续步骤使用。
SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)



# def parse_string_to_result(input_string: str):
import re, ast

def extract_dictionary_from_string(input_string: str):
    """
    Parse the model output into a dict keyed by int:
      {1: {"Reason": str, "Action": dict}, ...}

    Strategy:
      1) Normalize smart quotes / NBSP.
      2) Extract first "{" .. last "}" substring.
      3) Try json.loads (strict JSON from upgraded prompt).
      4) If json.loads fails, fall back to a more tolerant ast.literal_eval-based parser
         that attempts to quote unquoted keys and safely repr string fields.

    On unrecoverable parsing errors, return None instead of raising.
    """
    s = input_string

    # 1) Normalize smart quotes / weird spaces
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00A0", " ")

    # 2) Extract the {...} span (first "{" to last "}")
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return None
    d = s[start:end+1]

    # 3) Try strict JSON first (handles properly escaped internal quotes)
    try:
        parsed_json = json.loads(d)
        parsed = parsed_json
    except Exception:
        # 4) Fallback tolerant parser (based on original approach, safer for messy outputs)
        # Quote unquoted keys that start with a letter/underscore (leave numeric keys)
        d2 = re.sub(
            r'(?P<prefix>(?<=\{)|(?<=,)|(?<=\s))(?P<key>[A-Za-z_]\w*)\s*:',
            lambda m: m.group('prefix') + "'" + m.group('key') + "':",
            d
        )

        # Reason handler: normalise/quote Reason values safely
        reason_pattern = re.compile(r"('Reason'\s*:\s*)(.*?)(\s*,\s*'Action'\s*:)", re.DOTALL)
        def wrap_reason(m):
            prefix = m.group(1)
            val = m.group(2).strip()
            suffix = m.group(3)

            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                try:
                    parsed_str = ast.literal_eval(val)
                    return f"{prefix}{repr(parsed_str)}{suffix}"
                except Exception:
                    raw = val[1:-1]
                    raw = raw.replace("\\", "\\\\")
                    raw = raw.replace('"', '\\"')
                    return f"{prefix}{repr(raw)}{suffix}"

            return f"{prefix}{repr(val)}{suffix}"

        d2 = reason_pattern.sub(wrap_reason, d2)

        # text handler for Action->text values (handles unescaped inner quotes)
        text_pattern = re.compile(r"('text'\s*:\s*)(['\"])(.*?)\2(?=\s*(,|\}))", re.DOTALL)
        def wrap_text(m):
            prefix = m.group(1)
            val = m.group(3)
            literal = '"' + val.replace('\\"', '"').replace('"', '\\"') + '"'
            try:
                parsed = ast.literal_eval(literal)
                return f"{prefix}{repr(parsed)}"
            except Exception:
                raw = val.replace("\\", "\\\\")
                return f"{prefix}{repr(raw)}"
        d2 = text_pattern.sub(wrap_text, d2)

        # Generic handler for other string-like values of form : "..." or : '...'
        generic_pattern = re.compile(r"(:\s*)(['\"])(.*?)\2(?=\s*(,|\}|\]))", re.DOTALL)
        def wrap_generic(m):
            prefix = m.group(1)
            val = m.group(3)
            lit = '"' + val.replace('\\"', '"').replace('"', '\\"') + '"'
            try:
                parsed = ast.literal_eval(lit)
                return f"{prefix}{repr(parsed)}"
            except Exception:
                raw = val.replace("\\", "\\\\")
                return f"{prefix}{repr(raw)}"
        d2 = generic_pattern.sub(wrap_generic, d2)

        # Finally evaluate with ast.literal_eval
        try:
            parsed = ast.literal_eval(d2)
        except Exception:
            # Unrecoverable fallback failure -> return None
            return None

    # At this point `parsed` is a dict-like (keys might be str or int)
    if not isinstance(parsed, dict):
        return None

    # Build result structure and apply special rule
    result = {}
    for key, value in parsed.items():
        print(f"key:{key}, value:{value}")
        # keys might already be ints or strings; coerce to int
        try:
            int_key = int(key)
        except Exception:
            return None

        if not isinstance(value, dict) or "Reason" not in value or "Action" not in value:
            return None

        reason = value["Reason"]
        # ensure Reason is string
        if reason is None:
            reason = ""
        else:
            reason = str(reason)

        action = value["Action"] if isinstance(value["Action"], dict) else {}
        confidence = value["Confidence"]

        result[int_key] = {
            "Reason": reason,
            "Confidence": confidence,
            "Action": action
        }

    return result


# 为单个 UI 元素生成格式化的描述字符串。
# 输入UI元素对象和索引，输出格式化的字符串描述，包含元素的文本、内容描述、提示文本、工具提示、点击性、长按性、可编辑性、可滚动性、可聚焦性、选中性、检查性等信息，类似json
# 本质上就是以json格式文本化输出一个UIElement对象
def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Generate a description for a given UI element with important information.

  Args:
    ui_element: UI elements for the current screen.
    index: The numeric index for the UI element.

  Returns:
    The description for the UI element.
  """
  element_description = f'UI element {index}: {{"index": {index}, ' 
  if ui_element.text:
    element_description += f'"text": "{ui_element.text}", ' 
  if ui_element.content_description:
    element_description += (
        f'"content_description": "{ui_element.content_description}", '
    )
  if ui_element.hint_text:
    element_description += f'"hint_text": "{ui_element.hint_text}", ' 
  if ui_element.tooltip:
    element_description += f'"tooltip": "{ui_element.tooltip}", ' 
  element_description += (
      f'"is_clickable": {{"True" if ui_element.is_clickable else "False"}}, '
  )
  element_description += (
      '"is_long_clickable":'
      f' {{"True" if ui_element.is_long_clickable else "False"}}, '
  )
  element_description += (
      f'"is_editable": {{"True" if ui_element.is_editable else "False"}}, '
  )
  if ui_element.is_scrollable:
    element_description += '"is_scrollable": True, ' 
  if ui_element.is_focusable:
    element_description += '"is_focusable": True, ' 
  element_description += (
      f'"is_selected": {{"True" if ui_element.is_selected else "False"}}, '
  )
  element_description += (
      f'"is_checked": {{"True" if ui_element.is_checked else "False"}}, '
  )
  return element_description[:-2] + '}'

# 为整个 UI 元素列表生成描述字符串。
# 输入UI元素列表和屏幕尺寸，输出格式化的字符串描述，每个元素占一行
def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate concise information for a list of UIElement.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: The height and width of the screen in pixels.

  Returns:
    Concise information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += _generate_ui_element_description(ui_element, index) + '\n'
  return tree_info

# 生成动作选择阶段的完整提示。使用 {} 占位符以便动态填充：
# - {goal} : 用户当前目标
# - {history} : 已执行步骤的历史记录
# - {ui_elements} : 当前屏幕的 UI 元素列表
# - {additional_guidelines} : 任务特定的额外指导
def _action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
    additional_guidelines: list[str] | None = None,
    k=3,
) -> str:
  """Generate the prompt for the action selection.

  Args:
    goal: The current goal.
    history: Summaries for previous steps.
    ui_elements: A list of descriptions for the UI elements.
    additional_guidelines: Task specific guidelines.

  Returns:
    The text prompt for action selection that will be sent to gpt4v.
  """
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return ACTION_SELECTION_PROMPT_TEMPLATE.format(
      goal=goal,
      history=history,
      ui_elements=ui_elements if ui_elements else 'Not available',
      additional_guidelines=extra_guidelines,
      # k=k,
  )

# 生成步骤总结阶段的最终提示字符串。
# - action : 已执行的动作
# - reason : 执行动作的原因
# - goal : 整体目标
# - before_elements : 动作前的 UI 元素
# - after_elements : 动作后的 UI 元素
def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
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
  return SUMMARY_PROMPT_TEMPLATE.format(
      goal=goal,
      before_elements=before_elements,
      after_elements=after_elements,
      action=action,
      reason=reason,
  )


class M3A_VM(base_agent.EnvironmentInteractingAgent):
  """M3A which stands for Multimodal Autonomous Agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = 'M3A_WM',
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes a M3A Agent.

    Args:
      env: The environment.
      llm: The multimodal LLM wrapper.
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stablize
        after executing an action
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    self.history = []

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    # 创建字典存储当前步骤的所有数据
    step_data = {
        'raw_screenshot': None,
        'before_screenshot_with_som': None,
        'before_ui_elements': [],
        'after_screenshot_with_som': None,
        'origin_output': None,  # agent原本的输出
        # 'mllm_predict_html': None,  # mllm预测的gui的html代码
        'predicted_gui': None,  # 预测的下一时刻的gui图像
        'action_prompt': None,
        'action_output': None,
        'action_output_json': None,
        'action_reason': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    # 打印当前是第几步
    step_index=str(len(self.history) + 1)
    print('----------step ' + step_index)


    if len(self.history) + 1 >= 45:
      print("agent运行了超过45步，长难任务放弃")
      step_data['summary'] = 'The step is too long. Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    ################################# 获取当前状态和环境信息并填充进step_data #################################
    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    step_data['before_ui_elements'] = before_ui_elements
    before_ui_elements_list = _generate_ui_elements_description_list(
        before_ui_elements, logical_screen_size
    )
    step_data['raw_screenshot'] = state.pixels.copy()
    before_screenshot = state.pixels.copy()
    # 利用m3a_utils生成带标记的截图
    for index, ui_element in enumerate(before_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
    step_data['before_screenshot_with_som'] = before_screenshot.copy()
    # 生成动作选择prompt
    action_prompt = _action_selection_prompt(
        goal,
        [
            'Step ' + str(i + 1) + '- ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ],
        before_ui_elements_list,
        self.additional_guidelines,
    )
    step_data['action_prompt'] = action_prompt
    # print("当前步的提示词为：", action_prompt)
    ################################# mllm预测一步action及后处理 #################################
    action_output_temp, is_safe, raw_response = self.llm.predict_mm(  # 有空看看图像和prompt的关系，即如何插入图像的
        action_prompt,
        [
            step_data['raw_screenshot'],  # 原始屏幕截图
            before_screenshot,  # 带标记的截图
        ],
    )

    print("模型输出的内容为：", action_output_temp)

    # 如果该操作不安全，action_output修正为安全警报
    if is_safe == False:  # pylint: disable=singleton-comparison
      #  is_safe could be None
      action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
        Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
      
    # 如果没有输出，则说明LLM调用失败
    if not raw_response:
      raise RuntimeError('Error calling LLM in action selection phase.')

    ################################# World Model修正 #################################

    step_data['origin_output'] = action_output_temp
    actions_reason = extract_dictionary_from_string(action_output_temp)
    print("actions_reason:", actions_reason)

    action_sums = {}
    agent_sum = vimo_sum.T3A(self.env, infer.Gpt4Wrapper('gemini-2.5-flash'))
    agent_reward = vimo_reward.T3A(self.env, infer.Gpt4Wrapper('gemini-2.5-flash'))

    # 解析失败，该step直接结束
    if actions_reason is None:
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with required parameters (if any) in the'
          ' correct JSON format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    if len(actions_reason)<3 :
      print("模型只预测出了一个结果，不需要world model")
      reason = actions_reason[1]['Reason']
      action = actions_reason[1]['Action']
      action_output = f'Reason: {reason}\nAction: {action}'

    if actions_reason[1]['Action']['action_type'] in ["answer", "status"]:
      print(f"模型预测的第一个action为{actions_reason[1]['Action']['action_type']}，直接选择，跳过world model预测")
      reason = actions_reason[1]['Reason']
      action = actions_reason[1]['Action']
      action_output = f'Reason: {reason}\nAction: {action}'


    else:


      # 将模型输出的3个候选action逐个取出，用agent_sum生成用户行为描述，存进action_sums中
      cnt=0
      for act_key in actions_reason.keys():
        reason = actions_reason[act_key]['Reason']
        if actions_reason[act_key]['Confidence']==1:
          cnt+=1
        action = actions_reason[act_key]['Action']
        action_sum = agent_sum.step(goal, reason, action, before_ui_elements_list)
        action_sums[act_key] = action_sum.strip('\n')
      # 如果置信度为1的有且仅有一个，那么直接取第一个
      if cnt==1 or actions_reason[1]['Action']==actions_reason[2]['Action']:
        print("有且仅有一个action置信度为1，或者模型预测出两个相同的action，直接采用，不需要world model")
        reason = actions_reason[1]['Reason']
        action = actions_reason[1]['Action']
        action_output = f'Reason: {reason}\nAction: {action}'

      else:

        start_time = time.time()
        html_save_path=f"/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/html_tmp/{self.name}/{goal.split()[0]}/{step_index}/"
        os.makedirs(html_save_path, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(post_figure_action, step_data['raw_screenshot'], action_sums[1], html_save_path, 1),
                executor.submit(post_figure_action, step_data['raw_screenshot'], action_sums[2], html_save_path, 2),
                executor.submit(post_figure_action, step_data['raw_screenshot'], action_sums[3], html_save_path, 3),
            ]
        results = [future.result() for future in futures]
        step_data['predicted_gui'] = results

        end_time = time.time()
        print(f"gpt5根据3个action预测对应的下一时刻的gui所用时为: {end_time - start_time:.4f} s")


        flag=True

        # 如果有一个渲染失败了，就直接选第一个action。（如果将渲染失败的action舍弃，这并不合理）
        for i, arr in enumerate(results, start=1):
          if arr is None:
              flag=False

        if flag:
          # 可调整的默认分数（在 [-1,1] 范围内选一个合适的默认值）
          DEFAULT_SCORE = -1.0

          def _score_one_action(i, arr, default_score=DEFAULT_SCORE):
              """
              调用 agent_reward.step 并捕获异常。
              返回 (i, score) 对。发生异常或返回 None 时使用 default_score。
              """
              try:
                  score = agent_reward.step(
                      goal,
                      self.history,
                      actions_reason[i]['Action'],
                      actions_reason[i]['Reason'],
                      action_sums[i],
                      before_ui_elements_list,
                      before_screenshot,
                      arr
                  )
                  if score is None:
                      print("agent_reward.step returned None for action %s; using default %s", i, default_score)
                      score = default_score
                  return i, score
              except Exception as e:
                  # 记录完整堆栈以便排查，但不要抛出异常
                  print("Exception when scoring action %s: %s", i, e)
                  return i, default_score

          # 并发执行（线程数取 min(3, len(results)) 以防 results 数量变化）
          ac_dict = {}
          max_workers = min(3, max(1, len(results)))
          with ThreadPoolExecutor(max_workers=max_workers) as executor:
              futures = [executor.submit(_score_one_action, i, arr) for i, arr in enumerate(results, start=1)]
              for fut in as_completed(futures):
                  try:
                      i, score = fut.result()
                  except Exception as e:
                      # 这里理论上不会走到（因为 _score_one_action 已捕获所有异常），但仍作防护
                      print("Unexpected exception retrieving future result: %s", e)
                      # 找不到对应的索引无法回填，就跳过
                      continue
                  ac_dict[i] = score


          # 根据打分为三个action进行排序
          top_confidence_keys = sorted(
                          ac_dict.keys(),
                          key=lambda k: ac_dict[k],
                          reverse=True
                      )
          # 选择得分最高的
          selected_re = actions_reason[top_confidence_keys[0]]['Reason']
          selected_act = actions_reason[top_confidence_keys[0]]['Action']
          action_output = f'Reason: {selected_re}\nAction: {selected_act}'


          step_data['action_output'] = action_output
          step_data['action_raw_response'] = raw_response

        # 如果html预测失败，直接选第一个
        else:
          reason = actions_reason[0]['Reason']
          action = actions_reason[0]['Action']
          action_output = f'Reason: {reason}\nAction: {action}'

    ################################# Vimo修正完成 #################################

    # 解析最终的action和reason
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    step_data['action_reason'] = reason

    # 如果输出格式不正确，解析失败，并记录一个运行失败的summary并return结果
    if (not reason) or (not action):
      print('Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    # 结果解析成功
    print('Action: ' + action)
    print('Reason: ' + reason)

    # 尝试将action解析为json格式，将action转换为converted_action
    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
      step_data['action_output_json'] = converted_action
    # json解析失败，则记录一个运行失败的summary并return结果
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with required parameters (if any) in the'
          ' correct JSON format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    ################################# 执行动作 #################################
    # 获取action对应的操作的元素序号
    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)  # 可操作的总元素数量
    # 如果是点击，长按，输入文本，滚动操作，且有操作ui序号
    if (
        converted_action.action_type
        in ['click', 'long_press', 'input_text', 'scroll']
        and action_index is not None
    ):
      # 如果操作序号超出可操作元素范围，则异常，记录一个运行失败的summary并return结果
      if action_index >= num_ui_elements:
        print(
            f'Index out of range, prediction index is {action_index}, but the'
            f' UI element list only has {num_ui_elements} elements.'
        )
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

      # 否则操作序号合法，在raw_screenshot上添加操作ui元素的标记
      m3a_utils.add_ui_element_mark(
          step_data['raw_screenshot'],
          before_ui_elements[action_index],
          action_index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )

    # 如果返回的不是操作而是状态
    if converted_action.action_type == 'status':
      # 如果状态为不可行，打印日志
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      # 否则该状态就是任务成功，并返回结果
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    # 如果返回的是回答，直接打印输出
    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    # 尝试直接执行该动作
    try:
      self.env.execute_action(converted_action)
    # 如果报错，则执行失败，记录失败summary并返回结果
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to execute action.')
      print(str(e))
      step_data['summary'] = (
          'Can not execute the action, make sure to select the action with'
          ' the required parameters (if any) in the correct JSON format!'
      )
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    # 等待屏幕稳定
    time.sleep(self.wait_after_action_seconds)

    ################################# 记录执行后的环境状态和其他信息 #################################
    state = self.env.get_state(wait_to_stabilize=False)
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    after_ui_elements = state.ui_elements
    after_ui_elements_list = _generate_ui_elements_description_list(
        after_ui_elements, logical_screen_size
    )
    after_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            after_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    m3a_utils.add_screenshot_label(
        step_data['before_screenshot_with_som'], 'before'
    )
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    step_data['after_screenshot_with_som'] = after_screenshot.copy()

    ################################# 对前面的步骤进行总结摘要 #################################
    # 填充步骤总结prompt
    summary_prompt = _summarize_prompt(
        action,
        reason,
        goal,
        before_ui_elements_list,
        after_ui_elements_list,
    )
    # 生成总结摘要
    summary, is_safe, raw_response = self.llm.predict_mm(
        summary_prompt,
        [
            before_screenshot,
            after_screenshot,
        ],
    )
    # 如果LLM判断该步骤不安全，则记录unsafe summary并返回结果
    if is_safe == False:  # pylint: disable=singleton-comparison
      #  is_safe could be None
      summary = """Summary triggered LLM safety classifier."""

    # 如果没有回答，则调用llm生成总结失败，记录summary，返回结果
    if not raw_response:
      print(
          'Error calling LLM in summarization phase. This should not happen: '
          f'{summary}'
      )
      step_data['summary'] = (
          'Some error occurred calling LLM during summarization phase: %s'
          % summary
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    ################################# step_data填充完所有信息并返回最终结果 #################################
    step_data['summary_prompt'] = summary_prompt
    step_data['summary'] = f'Action selected: {action}. {summary}'
    print('Summary: ' + summary)
    step_data['summary_raw_response'] = raw_response

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
