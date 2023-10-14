import customtkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import cv2
import time
from ultralytics import YOLO
import numpy as np
from tensorflow import keras
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

root = tk.CTk()

model = torch.load("mine.pth")
model.to("cpu")
class Window:
    def __init__(self, root, classifier):
        self.classifier = classifier
        self.root = root
        self.objdet_model = YOLO("best.pt")
        self.cap = cv2.VideoCapture(0)
        self.root.title("App")
        self.root.geometry("1080x960")
        self.classes = ['Heart','Oblong', 'Oval', 'Round', 'Square']
        self.roi = None
        self.left_frame = tk.CTkFrame(root, bg_color="#0F0F0F")
        self.left_frame.grid(row=0, 
                             column=0, 
                             padx=10, 
                             pady=10, 
                             sticky="nsew")

        self.right_frame = tk.CTkFrame(root)
        self.right_frame.grid(row=0, 
                              column=1, 
                              padx=10, 
                              pady=10, 
                              sticky="nsew")


        self.left_canvas = tk.CTkCanvas(self.left_frame,
                                         width=500, 
                                         height=500)
        self.left_canvas.pack()
        self.blank_image = ImageTk.PhotoImage(image=Image.new("RGB", (500, 500)))
        self.cam = self.left_canvas.create_image(250,250,image=self.blank_image)
        
        self.right_canvas = tk.CTkCanvas(self.right_frame, 
                                         width=500, 
                                         height=500)
        self.right_canvas.pack()
        self.current = self.right_canvas.create_image(250,250,image=self.blank_image)

        main_frame = tk.CTkFrame(root)
        main_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.main_canvas = tk.CTkCanvas(main_frame, width=300, height=300)
        self.main_canvas.grid(column=1, row = 0)
        self.replacer= self.main_canvas.create_image(150, 150,image=self.blank_image)

        button = tk.CTkButton(main_frame, text="Create", width = 200, height= 100, command=self.screenshot)
        button.grid(column=0, row = 1)
        
        self.face_type = tk.CTkLabel(text="Face Type", 
                                     master=main_frame, 
                                     pady=100, 
                                     padx=200, 
                                     width=200, 
                                     height=200, 
                                     font = ("Times New Roman", 20))
        self.face_type.grid(column=0,
                             row =0)
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        self.update()
    def update(self):
        ret,frame = self.cap.read()
        if ret:
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.objdet_model(frame, verbose = False)
            boxes = result[0].boxes.cpu().numpy()
            if np.any(boxes):
                xyxy = (result[0].boxes.xyxy)[0].cpu().numpy()
                cv2.rectangle(frame, (int(xyxy[0]),int(xyxy[1])) ,(int(xyxy[2]),int(xyxy[3])), (0, 255, 0))
                self.roi = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame).resize((500,500), Image.Resampling.LANCZOS))

            self.left_canvas.itemconfig(self.cam ,image=photo)
            self.left_canvas.photo = photo
        self.root.after(10, self.update)
            
    def screenshot(self):
        self.real = Image.fromarray(self.roi)
        self.shot = ImageTk.PhotoImage(image=self.real.resize((500,500), Image.Resampling.LANCZOS))
        self.right_canvas.itemconfig(self.current ,image=self.shot)
        self.right_canvas.shot = self.shot
        self.classifier_func()

    def classifier_func(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        prod = transform(self.real)
        prod = prod.unsqueeze(0)
        prod = prod.type(torch.FloatTensor)
        output = model(prod)
        predicted_class = torch.argmax(output).item()
        self.output = self.classes[int(predicted_class)]
        print(self.output)
        self.face_type.configure(text = self.output)
        self.image_replacer()
    
    def image_replacer(self):
        result = Image.open(f"images/{self.output.capitalize()}.png").resize((300,300),Image.Resampling.LANCZOS)
        out = ImageTk.PhotoImage(image=result)
        self.main_canvas.itemconfig(self.replacer, image = out)
        self.main_canvas.out = out



    


app = Window(root, model)
root.mainloop()

app.cap.release()