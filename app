import tkinter as tk
from PIL import Image,ImageGrab
import torch
import numpy as np
from model_cnn import CNN

root = tk.Tk()
root.title('mnist')
root.geometry('700x600')

last_x = None
last_y = None

model = CNN().to('cpu')
model.load_state_dict(torch.load("mnist_cnn_model.pth"))
model.eval()

def start_draw(event):
    global last_x,last_y
    last_x = event.x
    last_y = event.y
    
def draw(event):
    if event.x % 10 == 0:
        print(event.x,event.y)
    global coord_label,last_x,last_y
    can.create_line(
        last_x,last_y,event.x,event.y,
        width = 5,
        fill = "black"
    )
    last_x = event.x
    last_y = event.y

def pos():
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
    
    
    with torch.no_grad():
        output = model(img)
        predicted = output.argmax(dim=1)
    
    print(x1,y1,x2,y2)
    print(f'预测结果：{predicted.item()}')
    
lab = tk.Label(root,text = 'please draw on it',font = ('Arial',12),bg = 'white',width = 15,height = 2)
lab.pack()

can = tk.Canvas(root,width=600,height=600,bg = 'white',bd = 2,relief = 'solid')
can.pack()

btn = tk.Button(root,text = 'submit',font = ('Arial',12),bg = 'white',width = 15,height = 2,command = pos)
btn.pack()

btn = tk.Button(root,text = 'clear',font = ('Arial',12),bg = 'white',width = 15,height = 2,command = lambda:can.delete("all"))
btn.pack()

root.update()
needed_height = root.winfo_reqheight()
root.geometry(f'800x{needed_height}')

can.bind("<Button-1>",start_draw)
can.bind("<B1-Motion>",draw)

root.mainloop()