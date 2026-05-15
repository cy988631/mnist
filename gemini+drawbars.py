import tkinter as tk
from PIL import Image, ImageGrab
import torch
import torch.nn.functional as F
import numpy as np
from model_cnn import CNN

# --- 模型加载 ---
model = CNN().to('cpu')
model.load_state_dict(torch.load("mnist_CNN_model.pth", map_location='cpu'))
model.eval()

root = tk.Tk()
root.title('MNIST 实时识别 - 极速版')
root.geometry('900x700')

last_x, last_y = None, None

# --- 推理与更新 UI ---
def update_prediction():
    # # 1. 截取画布区域
    # x1 = can.winfo_rootx()
    # y1 = can.winfo_rooty()
    # x2 = x1 + can.winfo_width()
    # y2 = y1 + can.winfo_height()

    # img = ImageGrab.grab((x1, y1, x2, y2)).convert("L").resize((28, 28))
    
    # # 2. 预处理
    # img_np = np.array(img)
    # img_np = 255 - img_np  # 反转：黑底白字
    # img_np = np.clip(img_np * 3, 0, 255).astype(np.uint8) # 增强对比度
    
    # tensor_img = torch.tensor(img_np/255.0).float().view(1, 1, 28, 28)
    # tensor_img = (tensor_img - 0.1307) / 0.3081

    x1 = can.winfo_rootx() + 100
    y1 = can.winfo_rooty() + 100
    x2 = x1 + 600
    y2 = y1 + 600

    img = ImageGrab.grab((x1, y1, x2, y2))
    img = img.convert("L")
    img = img.resize((28,28))
    # img = img.point(lambda x:0 if x < 128 else 255)
    # img.save('debug.png')
    
    img = np.array(img)
    img = 255-img
    img = np.clip(img * 3, 0, 255).astype(np.uint8)
    # img[img < 180] = 0
    # img[img >= 180] = 255
    Image.fromarray(img.astype(np.uint8)).save('debug.png')
    img = img/255.0
    img = (img-0.1307)/0.3081
    
    img = torch.tensor(img).float()
    img = img.view(1, 1, 28, 28)
    # 3. 模型推理
    with torch.no_grad():
        output = model(img)
        # 数学逻辑：Softmax 得到概率
        probs = F.softmax(output[0], dim=0).numpy()

    # 4. 更新可视化柱状图
    draw_bars(probs)
    predicted_num = np.argmax(probs)
    lab.config(text=f"预测结果: {predicted_num} (置信度: {probs[predicted_num]:.2%})")

# --- 绘制概率柱状图 ---
def draw_bars(probs):
    bar_can.delete("bar") # 清除旧的柱子
    max_h = 200 # 柱子最大高度
    for i, p in enumerate(probs):
        color = "green" if p == max(probs) else "gray"
        # 计算坐标 (每个柱子宽30，间距10)
        x_start = i * 40 + 10
        y_end = 220
        y_start = y_end - (p * max_h)
        
        # 画矩形
        bar_can.create_rectangle(x_start, y_start, x_start + 30, y_end, fill=color, tags="bar")
        # 画数字标签
        bar_can.create_text(x_start + 15, 235, text=str(i), tags="bar")

# --- 鼠标事件 ---
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    can.create_line(last_x, last_y, event.x, event.y, width=15, fill="black", capstyle=tk.ROUND, smooth=True)
    last_x, last_y = event.x, event.y
    update_prediction() # <--- 核心：画一笔，预测一次

def clear_all():
    can.delete("all")
    bar_can.delete("bar")
    lab.config(text="请在下方画板写字")

# --- UI 布局 ---
lab = tk.Label(root, text='请在下方画板写字', font=('Arial', 14))
lab.pack(pady=10)

# 侧边栏：放置画板和统计图
frame = tk.Frame(root)
frame.pack()

# 左侧画板
can = tk.Canvas(frame, width=400, height=400, bg='white', bd=2, relief='solid')
can.grid(row=0, column=0, padx=20)

# 右侧柱状图画布
bar_can = tk.Canvas(frame, width=420, height=250, bg='#F0F0F0')
bar_can.grid(row=0, column=1)

# 按钮
btn_clear = tk.Button(root, text='清除画板', command=clear_all, width=15)
btn_clear.pack(pady=20)

can.bind("<Button-1>", start_draw)
can.bind("<B1-Motion>", draw)

root.mainloop()