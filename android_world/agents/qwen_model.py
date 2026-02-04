import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from android_world.agents.wm_utils import render_aligned_png

class QwenModel:
    def __init__(self, model_name, device_map="auto"):
        """
        初始化 Qwen 模型类
        :param model_name: 模型路径
        :param device_map: 设备映射，默认为 "auto"
        """
        # 基础配置
        self.model_name = model_name
        self.device_map = device_map
        
        # 内部 prompt 模板
        self.system_prompt = "You are an expert **UI State Transition Simulator** and **Frontend Developer**.\nYour task is to predict the **NEXT UI STATE** based on a screenshot of the current state and a user interaction.\n\n### 1. IMAGE INTERPRETATION RULES\nThe input image contains visual cues denoting the user's action. You must interpret them as follows:\n*   **Red Circle**: Indicates a **Click** or **Long Press** target at that location.\n*   **Red Arrow**: Indicates a **Scroll** or **Swipe**.\n    *   The arrow points in the direction of finger movement.\n    *   *Example*: An arrow pointing UP means the finger slides up, pushing content up (Scrolling Down).\n*   **Note**: These cues exist ONLY to show the action. **DO NOT render these red circles or arrows in your output HTML.**\n\n### 2. CRITICAL STRUCTURAL RULES (MUST FOLLOW)\n*   **Format**: Output ONLY raw HTML. Start with `<!DOCTYPE html>` and end with `</html>`.\n*   **Root Element**: All visible content MUST be wrapped in:\n    `<div id=\"render-target\"> ... </div>`\n*   **Container Style**: `#render-target` must have:\n    `width: 1080px; height: 2400px; position: relative; overflow: hidden;`\n    (Apply background colors and shadows here, NOT on the body).\n*   **Body Style**: The `<body>` tag must have `margin: 0; padding: 0; background: transparent;`.\n*   **Layout**: Do NOT center the body. Let `#render-target` sit at (0,0).\n\n### 3. CONTENT GENERATION LOGIC\n*   **Transition**: Analyze the action. If the user clicks a button, show the *result* (e.g., a menu opens, a checkbox checks, page navigates).\n*   **Images**: Use semantic text placeholders. DO NOT use real URLs.\n    *   Format: `<div style=\"...\">[IMG: description]</div>`\n*   **Icons**: Use simple inline SVG paths or Unicode.\n\n### 4. OUTPUT REQUIREMENT\n*   Do NOT generate Markdown blocks (```html).\n*   Do NOT provide explanations or conversational text.\n*   Output the code directly.\n"
        self.user_prompt_template = "<image>\n### INPUT CONTEXT\n1.  **User Intent**: {goal}\n2.  **Interaction Description**:\n    *   **Description**: {description}\n    *   **Action Data**: {action}\n\n### COMMAND\nBased on the visual cues in the image and the interaction data above, generate the **HTML for the RESULTING UI STATE** (what the screen looks like *after* this action).\n"

        
        # 加载模型和处理器
        print(f"加载模型中: {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )
        print("模型加载完毕")
    
    def extract_clean_html(self, text):
        """
        清洗输出：
        1. 去除 ```html ... ```
        2. 提取 <!DOCTYPE html>...</html>
        :param text: 模型生成的文本
        :return: 清洗后的纯HTML文本
        """
        text = text.replace("```html", "").replace("```", "")
        
        start_match = re.search(r"<!DOCTYPE html>", text, re.IGNORECASE)
        end_match = re.search(r"</html>", text, re.IGNORECASE)
        
        if start_match and end_match:
            start_idx = start_match.start()
            end_idx = end_match.end()
            if end_idx > start_idx:
                return text[start_idx:end_idx]
                
        return text.strip()
    
    def post(self, goal, description, image, action, html_save_path, idx):
        """
        处理请求，生成HTML并渲染为图像
        :param image: PIL图像对象
        :param action: 用于填充模板的变量
        :param html_save_path: HTML文件保存路径
        :param idx: 索引，用于文件名
        :return: 渲染后的图像数组
        """
        # 填充用户prompt模板
        user_text = self.user_prompt_template.format(goal=goal, description=description, action=str(action))
        print("user_text:", user_text)
        # 构建Messages
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        
        try:
            # print("填写prompt模板")
            # 应用模板与编码
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            # print("填写prompt模板完毕")
            # print("使用设备：", self.model.device)
            
            # 将输入移动到模型设备
            inputs = inputs.to(self.model.device)

            print("开始生成html")
            # 生成回复
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            print("html生成完毕")
            # 解码
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            # print("解码完毕")
            # 清洗HTML
            clean_html = self.extract_clean_html(output_text)
            # print(clean_html)
            
            # 保存HTML文件
            html_path = os.path.join(html_save_path, f"test_{idx}.html")
            self.save_html(clean_html, html_path)
            
            # 渲染HTML为图像
            png_path = os.path.join(html_save_path, f"test_{idx}.png")
            success = render_aligned_png(html_path, png_path)
            
            if success:
                # 读取并返回图像数组
                from PIL import Image
                import numpy as np
                with Image.open(png_path) as img:
                    # 确保图像是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return np.array(img)
            else:
                print(f"[Error] 将html文件保存成图像失败： test_{idx}")
                return None
            
        except Exception as e:
            print(f"[Error] Qwen3推理失败，将返回None： {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_html(self, html_content, file_path):
        """
        保存HTML内容到文件
        :param html_content: HTML内容
        :param file_path: 保存路径
        :return: 是否保存成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"HTML saved successfully to {file_path}")
            return True
            
        except Exception as e:
            print(f"[Error] Save failed: {e}")
            return False
