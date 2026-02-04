import os
import math
from PIL import Image, ImageDraw

class SimpleVisualizer:
    def __init__(self):
        # 预定义颜色
        self.fill_color = (255, 0, 0, 100)    # 半透明红 (填充)
        self.outline_color = (255, 0, 0, 255) # 实心红 (边框)
        self.arrow_width = 15                 # 箭头线宽
        self.click_radius = 40                # 点击圆圈半径

    def visualize(self, image, save_path, action):
        """
        核心方法：绘制操作并保存。
        
        :param src_path:    输入图片路径
        :param save_path:   输出图片路径
        :param action_type: 'click' 或 'swipe' (也可以叫 'scroll')
        :param points:      坐标数据
                            - click: [x, y]
                            - swipe: [[start_x, start_y], [end_x, end_y]]
        """

        # 转换为 RGBA 以支持半透明绘制
        image = image.convert("RGBA")
        # 创建一个透明图层用于绘制
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        action_type = action['action']

        # --- 绘制逻辑 ---
        if action_type in ["click", "tap", "long_press"]:
            self._draw_click(draw, action['coordinate'])
        elif action_type in ["swipe", "scroll"]:
            self._draw_swipe(draw, [action['coordinate'], action['coordinate2']])
        else:
            print(f"⚠️ 未知动作类型: {action_type}")
            return

        # 合并图层并转换为 RGB (去除 Alpha 通道以便保存为 jpg/png)
        final_img = Image.alpha_composite(image, overlay).convert("RGB")
        
        # 自动创建父目录
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        final_img.save(save_path)
        print(f"✅ 已保存: {save_path}")

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

# ==========================================
# 使用方法
# ==========================================

if __name__ == "__main__":
    # 1. 实例化
    viz = SimpleVisualizer()

    # 假设你有一张图片叫 demo.png
    # 如果没有，请先放一张图片在当前目录，或者修改下面的路径
    input_img = "demo.png" 
    
    # === 示例 A: 点击 (Click) ===
    # 在坐标 (500, 800) 处画一个圈
    viz.visualize(
        src_path=input_img,
        save_path="output_click.png",
        action_type="click",
        points=[500, 800]
    )

    # === 示例 B: 滑动 (Swipe) ===
    # 从 (500, 1500) 滑动到 (500, 500) —— 模拟向上滑屏
    viz.visualize(
        src_path=input_img,
        save_path="output_swipe.png",
        action_type="swipe",
        points=[[500, 1500], [500, 500]]
    )