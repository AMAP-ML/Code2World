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

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import json
import os
import random
import time
from typing import Type
from PIL import Image
import numpy as np

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer
from android_world.agents import m3a, m3a_wm
# from android_world.agents import m3a_qwen
from android_world.env import env_launcher
from android_world.task_evals import task_eval

# CUDA_VISIBLE_DEVICES=1,2 



# 设置日志级别为 WARNING
logging.set_verbosity(logging.WARNING)

# 配置 GRPC 环境变量，仅显示错误信息并禁用跟踪
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


# 返回adb的路径
def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  # 先从用户目录下面找adb可能在的路径
  potential_paths = [
      os.path.expanduser('/home/zyh/zyh_documents/zla/bin/Android/Sdk/platform-tools/adb'),
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  # 没有找到adb则抛出环境错误
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )

# 设置命令行参数
# 设置ADB路径
_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),  # 返回默认路径
    'Path to adb. Set if not installed through SDK.',
)
# 是否执行模拟器设置（仅首次运行前需要）
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
# AVD的端口，默认为5554，在启动AVD时设置
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    # 5556,
    5558,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
# 指定任务名称，默认为None，随机选择一个任务
_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)


def _main() -> None:
  """Runs a single task."""
  # 加载并设置环境
  # 这一步似乎非常耗时
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,  # 端口
      emulator_setup=_EMULATOR_SETUP.value,  # 是否进行模拟器设置
      adb_path=_ADB_PATH.value,  # adb路径
      grpc_port=8564
  )
  # 重置环境，将界面返回到Home页面
  env.reset(go_home=True)
  # 获取任务注册表
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  # 如果指定了任务则使用指定任务，否则随机选择一个任务
  if _TASK.value:
    # 若不存在这个任务，弹出报错
    if _TASK.value not in aw_registry:
      raise ValueError('Task {} not found in registry.'.format(_TASK.value))
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
  else:
    # 随机选择一个任务
    task_type: Type[task_eval.TaskEval] = random.choice(
        list(aw_registry.values())
    )
  print(f"任务为：{task_type}")
  # 根据task_type随机生成一组任务参数
  params = task_type.generate_random_params()
  # 创建并初始化一个task_type的任务
  task = task_type(params)
  task.initialize_task(env)
  # 创建agent
  # agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-5'))
  agent = m3a.M3A(env, infer.Gpt4Wrapper('gemini-2.5-flash'))
  # agent = m3a_wm.M3A_VM(env, infer.Gpt4Wrapper('gemini-2.5-flash'))
  # agent = m3a_wm.M3A_VM(env, infer.Gpt4Wrapper('gemini-2.5-pro'))
  # agent = m3a_wm.M3A_VM(env, infer.Gpt4Wrapper('gpt-5'))
  # agent = m3a_qwen.M3A_VM(env, infer.Gpt4Wrapper('gemini-2.5-flash'))

  print('Goal: ' + str(task.goal))
  is_done = False
  
  # 创建实验日志文件夹
  experiment_dir = os.path.join(os.getcwd(), f'experiment_logs_{time.strftime("%Y%m%d_%H%M%S")}')
  os.makedirs(experiment_dir, exist_ok=True)
  
  # 执行任务，最多执行task.complexity * 10步
  for step_idx in range(int(task.complexity * 10)):
    response = agent.step(task.goal)
    

    # 保存数据
    if response.data:
      step_data = response.data
      
      # 为当前步骤创建单独的子文件夹
      step_dir = os.path.join(experiment_dir, f'step_{step_idx+1}')
      os.makedirs(step_dir, exist_ok=True)
      
      # 分离图片数据和其他数据
      image_data = {}
      json_data = {}
      
      for key, value in step_data.items():
        if key in ['before_screenshot', 'after_screenshot', 'raw_screenshot', 'before_screenshot_with_som', 'after_screenshot_with_som']:
          image_data[key] = value
        elif key in ['action_prompt', 'action_output', 'summary_prompt', 'summary', 'origin_output']:
          json_data[key] = value
        elif key in ['summary']:
          image_data[key] = value[0]

      # 保存图片
      for img_key, img_array in image_data.items():
        if img_array is not None:
          img = Image.fromarray(img_array)
          img_path = os.path.join(step_dir, f'{img_key}.png')
          img.save(img_path)
          print(f'Saved {img_key}: {img_path}')
      
      # 保存其他数据为json
      json_path = os.path.join(step_dir, 'data.json')
      with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
      print(f'Saved step data: {json_path}')
    
    if response.done:  # 如果任务完全则提前结束
      is_done = True
      break
  # agent任务完全且环境判断成功，则任务成功
  agent_successful = is_done and task.is_successful(env) == 1
  print(
      f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
      f' {task.goal}'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
