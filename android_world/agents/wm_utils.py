import io
import base64
import requests
import numpy as np
from typing import Optional, Any
import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import math
from PIL import Image, ImageDraw
from android_world.env import representation_utils
# 导入项目中已有的函数
from android_world.agents.infer import array_to_jpeg_bytes, image_to_jpeg_bytes

# 定义常量
ERROR_CALLING_LLM = "ERROR_CALLING_LLM"

# 系统指令
SYSTEM_PROMPT = """You are an expert **UI State Transition Simulator** and **Frontend Developer**.
Your task is to predict the **NEXT UI STATE** based on a screenshot of the current state and a user interaction.

### 1. IMAGE INTERPRETATION RULES
The input image contains visual cues denoting the user's action. You must interpret them as follows:
*   **Red Circle**: Indicates a **Click** or **Long Press** target at that location.
*   **Red Arrow**: Indicates a **Scroll** or **Swipe**.
    *   The arrow points in the direction of finger movement.
    *   *Example*: An arrow pointing UP means the finger slides up, pushing content up (Scrolling Down).
*   **Note**: These cues exist ONLY to show the action. **DO NOT render these red circles or arrows in your output HTML.**

### 2. CRITICAL STRUCTURAL RULES (MUST FOLLOW)
*   **Format**: Output ONLY raw HTML. Start with `<!DOCTYPE html>` and end with `</html>`.
*   **Root Element**: All visible content MUST be wrapped in:
    `<div id="render-target"> ... </div>`
*   **Container Style**: `#render-target` must have:
    `width: 1080px; height: 2400px; position: relative; overflow: hidden;`
    (Apply background colors and shadows here, NOT on the body).
*   **Body Style**: The `<body>` tag must have `margin: 0; padding: 0; background: transparent;`.
*   **Layout**: Do NOT center the body. Let `#render-target` sit at (0,0).

### 3. CONTENT GENERATION LOGIC
*   **Transition**: Analyze the action. If the user clicks a button, show the *result* (e.g., a menu opens, a checkbox checks, page navigates).
*   **Images**: Use semantic text placeholders. DO NOT use real URLs.
    *   Format: `<div style="...">[IMG: description]</div>`
*   **Icons**: Use simple inline SVG paths or Unicode.

### 4. OUTPUT REQUIREMENT
*   Do NOT generate Markdown blocks (```html).
*   Do NOT provide explanations or conversational text.
*   Output the code directly.
"""

# 用户指令，填充action
USER_PROMPT_TEMPLATE = """
### INPUT CONTEXT
1.  **User Intent**: "Interact with UI."
2.  **Interaction Details**:
    *   **Description**: {semantic_desc}

### COMMAND
Based on the image and the interaction data above, generate the **HTML for the RESULTING UI STATE** (what the screen looks like *after* this action).
"""



# 定义gpt5请求的类，参考infer.py的实现
class MLLMInferencer:
    def __init__(self, model_name="gpt-5", temperature=0.0, max_retry=3):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise RuntimeError('OpenAI API key not set.')
        self.model = model_name
        self.temperature = temperature
        self.max_retry = max_retry
        self.RETRY_WAITING_SECONDS = 20  
    
    def encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string using project's existing function."""
        return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')
    
    # 输入用户的prompt（包含action），当前gui图像，返回MLLM的输出（预期是纯html代码）
    def predict_mm(self, user_prompt: str, images: list[np.ndarray]) -> tuple[str, Optional[bool], Any]:
        """Send prompt and images to GPT-5 and get response, following the same format as infer.py."""
        headers = {
            'Content-Type': 'application/json',
            # 'Authorization': f'Bearer {self.openai_api_key}',  
            'Authorization': f'Bearer sk-LFjq1ToJQrknIHrrohUMA21z7iSsETOyhfUUuko6hJsRozXU', 
        }

        payload = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': SYSTEM_PROMPT},  # 系统提示，定义角色和任务
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_prompt},
                    ],
                }
            ],
        }

        # print("开始编码图像")
        for image in images:
            payload['messages'][1]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
                },
            })
        # print("图像编码完毕")

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                # print(f"开始请求{self.model}")
                response = requests.post(
                    # 'https://tao.plus7.plus/v1/chat/completions',  # 使用与项目中相同的API端点
                    'https://chat.intern-ai.org.cn/api/v1/chat/completions',  # 使用与项目中相同的API端点
                    headers=headers,
                    json=payload,
                )
                if response.ok and 'choices' in response.json():
                    # print("请求成功")
                    return (
                        response.json()['choices'][0]['message']['content'],
                        None,
                        response,
                    )
                print(
                    'Error calling OpenAI API with error message: '
                    + response.json()['error']['message']
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Catch all exceptions during LLM calls
                print("请求失败")
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
        return ERROR_CALLING_LLM, None, None


def render_aligned_png(html_path, output_path):
    """
    智能渲染 HTML 为 PNG。
    策略：
    1. 优先尝试全资源加载 (High Fidelity)。
    2. 如果超时，自动降级为拦截字体模式 (High Reliability)。
    """
    
    # --- 1. 尺寸获取 ---
    target_width, target_height = 1080, 2400 

    abs_html_path = os.path.abspath(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # 上下文配置 (一次性配置好视口)
        context = browser.new_context(
            viewport={'width': target_width, 'height': target_height},
            device_scale_factor=1.0
        )

        # ============================================================
        # 🟢 尝试 1: 高保真模式 (不拦截，等待字体加载)
        # ============================================================
        page = context.new_page()
        success = False
        
        try:
            # print(f"   🔄 尝试高保真渲染...")
            
            # 这里的 timeout 设为 30000ms (30秒)。
            # 如果 6秒内字体下不下来，说明网络大概率不行，不如赶紧切 Plan B
            page.goto(f"file://{abs_html_path}", wait_until="load", timeout=30000)
            
            # 等待网络空闲 (确保图片和字体都好了)
            # 稍微给一点缓冲
            page.wait_for_load_state("networkidle", timeout=2000)
            
            page.screenshot(path=output_path, omit_background=True)
            print(f"✔ [High-Fi] 已生成: {output_path}")
            success = True

        except PlaywrightTimeoutError:
            print(f"   ⚠️ 高保真加载超时 (字体/图片太慢)，切换到极速模式...")
        except Exception as e:
            print(f"   ⚠️ 高保真渲染出错: {e}，切换到极速模式...")
        finally:
            # 无论成功失败，先关掉这个页面，清理资源
            page.close()

        # ============================================================
        # 🔴 尝试 2: 极速兜底模式 (如果尝试 1 失败)
        # ============================================================
        if not success:
            page_fallback = context.new_page()
            try:
                # 1. 拦截字体和 Google 请求
                page_fallback.route("**/*.{woff,woff2,ttf,otf,eot}", lambda route: route.abort())
                page_fallback.route("**/*fonts.googleapis.com*", lambda route: route.abort())
                page_fallback.route("**/*fonts.gstatic.com*", lambda route: route.abort())
                
                # 2. 注入系统字体 CSS
                page_fallback.add_init_script("""
                    const style = document.createElement('style');
                    style.innerHTML = `
                        * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }
                    `;
                    document.head.appendChild(style);
                """)

                # 3. 快速加载 (只等 DOM)
                page_fallback.goto(f"file://{abs_html_path}", wait_until="domcontentloaded", timeout=15000)
                
                # 简单缓冲
                page_fallback.wait_for_timeout(500)
                
                page_fallback.screenshot(path=output_path, omit_background=True)
                print(f"✔ [Fallback] 已生成: {output_path}")
                
            except Exception as e:
                print(f"❌ 彻底失败 [{os.path.basename(html_path)}]: {e}")
            finally:
                page_fallback.close()

        browser.close()

        return success


# PNG图像与image_array的相互转换函数
def png_to_image_array(png_path: str) -> np.ndarray:
    """将PNG文件转换为NumPy数组格式的image_array
    
    Args:
        png_path: PNG图像文件的路径
        
    Returns:
        NumPy数组格式的图像数据
    """
    with Image.open(png_path) as img:
        # 确保图像是RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

def image_array_to_png(image_array: np.ndarray, output_path: Optional[str] = None) -> bytes:
    """将NumPy数组格式的image_array转换为PNG格式
    
    Args:
        image_array: NumPy数组格式的图像数据
        output_path: 可选的输出文件路径，如果提供则保存到文件
        
    Returns:
        PNG格式的图像字节数据
    """
    # 创建PIL图像对象
    img = Image.fromarray(image_array)
    
    # 创建字节流
    png_bytes = io.BytesIO()
    
    # 保存为PNG格式
    img.save(png_bytes, format='PNG')
    png_bytes = png_bytes.getvalue()
    
    # 如果提供了输出路径，则保存到文件
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
    
    return png_bytes


def android_world_to_control(android_world: dict, ui_elements: list[representation_utils.UIElement]) -> dict:
    """Convert android_world format to android_control format."""
    
    action_type = android_world.get("action_type")
    
    # AC的json格式初始化
    control_action = {"name": "mobile_use", "arguments": {}}
    
    # 如果是"status"状态类型，转成AC的"action": "terminate"
    if action_type == "status":
        goal_status = android_world.get("goal_status")
        if goal_status == "complete":
            control_action["arguments"]["action"] = "terminate"
            control_action["arguments"]["status"] = "success"
        elif goal_status == "infeasible":
            control_action["arguments"]["action"] = "terminate"
            control_action["arguments"]["status"] = "failure"
    
    # AC不支持answer
    elif action_type == "answer":
        text = android_world.get("text")
        control_action["arguments"]["action"] = "answer"
        control_action["arguments"]["text"] = text
    
    # 对于点击事件
    elif action_type == "click":
        index = android_world.get("index")  # 获取点击事件操作的element
        if index is not None and index != '':
            index = int(index)
        # 根据index从ui_elements中检索对应的元素
        print(index, len(ui_elements))
        if index is not None and 0 <= index < len(ui_elements):
            ui_element = ui_elements[index]
            if ui_element.bbox_pixels:
                center_x, center_y = ui_element.bbox_pixels.center
                control_action["arguments"]["action"] = "click"
                control_action["arguments"]["coordinate"] = [int(center_x), int(center_y)]
            else:
                print(f"index为{index},列表总长为{len(ui_elements)}, index不合法！！！")
    
    elif action_type == "long_press":
        index = android_world.get("index")
        if index is not None and index != '':
            index = int(index)
        if index is not None and 0 <= index < len(ui_elements):
            ui_element = ui_elements[index]
            if ui_element.bbox_pixels:
                center_x, center_y = ui_element.bbox_pixels.center
                control_action["arguments"]["action"] = "long_press"
                control_action["arguments"]["coordinate"] = [int(center_x), int(center_y)]
                # Default time if not provided
                control_action["arguments"]["time"] = 1.0
    
    elif action_type == "input_text":
        text = android_world.get("text")
        index = android_world.get("index")
        if text is not None:
            control_action["arguments"]["action"] = "type"
            control_action["arguments"]["text"] = text
            # Note: input_text in android_world includes clicking the field, but in android_control
            # we need separate click and type actions. However, the requirement says to convert to
            # android_control format, so we'll just do the type part here.
    
    elif action_type == "keyboard_enter":
        control_action["arguments"]["action"] = "key"
        control_action["arguments"]["key_code"] = "enter"
    
    elif action_type == "navigate_home":
        control_action["arguments"]["action"] = "system_button"
        control_action["arguments"]["button"] = "home"
    
    elif action_type == "navigate_back":
        control_action["arguments"]["action"] = "system_button"
        control_action["arguments"]["button"] = "back"
    
    elif action_type == "scroll":
        direction = android_world.get("direction")
        index = android_world.get("index")
        if index is not None and index != '':
            index = int(index)
        
        control_action["arguments"]["action"] = "swipe"
        
        # For scroll, we need to determine the swipe coordinates
        # If index is provided, use that element's center; otherwise, use screen center
        if index is not None and index != '' and 0 <= index < len(ui_elements):
            ui_element = ui_elements[index]
            if ui_element.bbox_pixels:
                center_x, center_y = ui_element.bbox_pixels.center
        else:
            # Default to screen center (assuming typical phone resolution)
            center_x, center_y = 540, 1200
        
        # Determine swipe coordinates based on direction
        if direction == "up":
            control_action["arguments"]["coordinate"] = [int(center_x), int(center_y + 200)]
            control_action["arguments"]["coordinate2"] = [int(center_x), int(center_y - 200)]
        elif direction == "down":
            control_action["arguments"]["coordinate"] = [int(center_x), int(center_y - 200)]
            control_action["arguments"]["coordinate2"] = [int(center_x), int(center_y + 200)]
        elif direction == "left":
            control_action["arguments"]["coordinate"] = [int(center_x + 200), int(center_y)]
            control_action["arguments"]["coordinate2"] = [int(center_x - 200), int(center_y)]
        elif direction == "right":
            control_action["arguments"]["coordinate"] = [int(center_x - 200), int(center_y)]
            control_action["arguments"]["coordinate2"] = [int(center_x + 200), int(center_y)]
    
    elif action_type == "open_app":
        app_name = android_world.get("app_name")
        if app_name is not None:
            control_action["arguments"]["action"] = "open"
            control_action["arguments"]["text"] = app_name
    
    elif action_type == "wait":
        control_action["arguments"]["action"] = "wait"
        control_action["arguments"]["time"] = 1.0  # Default wait time
    
    return control_action


class SimpleVisualizer:
    def __init__(self):
        # 预定义颜色
        self.fill_color = (255, 0, 0, 100)    # 半透明红 (填充)
        self.outline_color = (255, 0, 0, 255) # 实心红 (边框)
        self.arrow_width = 15                 # 箭头线宽
        self.click_radius = 40                # 点击圆圈半径

    def visualize(self, image, action):
        """
        核心方法：绘制操作并保存。
        
        :param src_path:    输入图片路径
        :param save_path:   输出图片路径
        :param action_type: 'click' 或 'swipe' (也可以叫 'scroll')
        :param points:      坐标数据
                            - click: [x, y]
                            - swipe: [[start_x, start_y], [end_x, end_y]]
        """
        # print("visualize中传入的action为：", action)
        # 转换为 RGBA 以支持半透明绘制
        image = image.convert("RGBA")
        # 创建一个透明图层用于绘制
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        action_type = action['action']

        # --- 绘制逻辑 ---
        if action_type in ["click", "tap", "long_press"]:
            print(f"✅ 动作类型为: {action_type}")
            self._draw_click(draw, action['coordinate'])
        elif action_type in ["scroll"]:
            print(f"✅ 动作类型为: {action_type}")
            self._draw_swipe(draw, [action['coordinate'], action['coordinate2']])
        else:
            print(f"⚠️ 未知动作类型: {action_type}")
            return image

        # 合并图层并转换为 RGB (去除 Alpha 通道以便保存为 jpg/png)
        final_img = Image.alpha_composite(image, overlay).convert("RGB")

        return final_img

    def _draw_click(self, draw, point):
        """画圆圈"""
        if not point or len(point) < 2: return
        cx, cy = point[0], point[1]
        r = self.click_radius
        # 绘制：(左上角x, 左上角y, 右下角x, 右下角y)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), 
                     fill=self.fill_color, outline=self.outline_color, width=5)

    def _draw_swipe(self, draw, points):
        """画箭头"""
        # points 应该是 [[x1, y1], [x2, y2]]
        if not points or len(points) < 2: return
        
        start = tuple(points[0]) # 起点
        end = tuple(points[1])   # 终点
        
        # 1. 画线
        draw.line([start, end], fill=self.outline_color, width=self.arrow_width)
        
        # 2. 画箭头头部 (数学计算)
        x1, y1 = start
        x2, y2 = end
        arrow_len = 50
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # 箭头两翼坐标
        p1_x = x2 - arrow_len * math.cos(angle - math.pi / 6)
        p1_y = y2 - arrow_len * math.sin(angle - math.pi / 6)
        p2_x = x2 - arrow_len * math.cos(angle + math.pi / 6)
        p2_y = y2 - arrow_len * math.sin(angle + math.pi / 6)
        
        # 绘制三角形箭头头
        draw.polygon([(x2, y2), (p1_x, p1_y), (p2_x, p2_y)], fill=self.outline_color)

viz = SimpleVisualizer()


# 修改后的post_figure_action函数
def _post_figure_action(inferencer, image_array, action, html_save_path, idx):

    # 1. 创建inferencer实例（API密钥从环境变量获取）    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        semantic_desc=action,
    )

    # 2. 调用gpt5 API获取HTML响应
    response_text, _, _ = inferencer.predict_mm(user_prompt, [image_array])
    
    # print("response_text:", response_text)

    if response_text == ERROR_CALLING_LLM:
        raise Exception("Error calling GPT-5 API")
    
    # 3. 确保响应是纯HTML格式
    # 简单检查响应是否包含HTML标签
    if not (response_text.strip().startswith('<') and response_text.strip().endswith('>')):
        print("mllm的输出格式不是纯粹的html格式！！！")
        pass
    
    # 4. 将mllm输出的HTML文本先保存成html文件，再渲染为图像
    html_path=html_save_path+f"test_{idx}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    image_path=html_save_path+f"test_{idx}.png"
    flag = render_aligned_png(html_path, image_path)
    # 渲染成功
    if flag:
    # 5.读取图像并存成numpy格式
        returned_array = png_to_image_array(image_path)
        print("成功保存图像到：", image_path)
        return returned_array
    else:
        return None

######################### 测试 #########################
if __name__ == '__main__':
    # 当前gui界面
    image_url="/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/demo_exp/experiment_logs_20251215_161736/step_1/before_screenshot.png"
    image_array=png_to_image_array(image_url)
    action="Open the app: Settings"

    inferencer = MLLMInferencer(model_name="gpt-5")

    _post_figure_action(inferencer, image_array=image_array, action=action)



    # print(inferencer.openai_api_key)

    # user_prompt = USER_PROMPT_TEMPLATE.format(
    #     semantic_desc=action,
    # )
    # print(user_prompt)

    # # 2. 调用gpt5 API获取HTML响应
    # response_text, _, _ = inferencer.predict_mm(user_prompt, [image_array])

    # print(response_text)

    # print("response_text:", response_text)

    # if response_text == ERROR_CALLING_LLM:
    #     raise Exception("Error calling GPT-5 API")

    # # 3. 确保响应是纯HTML格式
    # # 简单检查响应是否包含HTML标签
    # if not (response_text.strip().startswith('<') and response_text.strip().endswith('>')):
    #     # 如果不是纯HTML，可能需要提取HTML部分
    #     # 这里可以根据实际情况调整
    #     pass

    # html_path="test.html"
    # # 4. 将HTML响应渲染为图像
    # with open(html_path, "w", encoding="utf-8") as f:
    #     f.write(response_text)

    # render_aligned_png(html_path, "test.png")