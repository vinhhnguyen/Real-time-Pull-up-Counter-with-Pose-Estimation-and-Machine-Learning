import tkinter as tk 
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

window = tk.Tk()
window.title("Live Feed")
window.geometry("640x480")
ck.set_appearance_mode("dark")

frame = tk.Frame(window, width=640, height=480)
frame.place(x=0, y=0)
lmain = tk.Label(frame)

lmain.place(x=0, y=0)

classLabel = ck.CTkLabel(window, text="STAGE", height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)

counterLabel = ck.CTkLabel(window, text="COUNT", height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=170, y=1)

probLabel  = ck.CTkLabel(window, text="PROB", height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=330, y=1)

classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)

counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=170, y=41)

probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=330, y=41)

def reset_counter(): 
    global counter
    counter = 0 

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=340)

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

stage_position = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''

cap = cv2.VideoCapture(0)

def detect():
    global stage_position, counter, bodylang_prob, bodylang_class

    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    try: 
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0] 

        if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
            stage_position = "down" 
        elif stage_position == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            stage_position = "up" 
            counter += 1 

    except Exception as e:
        print(e) 

    img = image[:, :480, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(10, detect)  

    counterBox.configure(text=counter) 
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}")
    classBox.configure(text=stage_position) 


detect() 
window.mainloop()
