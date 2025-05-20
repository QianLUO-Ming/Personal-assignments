import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================== Fix 1: Optimized Model Initialization ==================
def load_emotion_model():
    # Create model structure
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    # Compile model (required step)
    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
    
    # Load weights
    try:
        model.load_weights('emotion_model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        exit()
    return model

# ================== Global Configurations ==================
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

emoji_dist = {
    0: "./emojis/angry.png",
    1: "./emojis/disgusted.png",
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png",
    4: "./emojis/neutral.png",
    5: "./emojis/sad.png",
    6: "./emojis/surprised.png"  
}

# ================== Main GUI Application ==================
class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Real-Time Emotion Detection")
        self.window.configure(bg='white')
        
        # Initialize model
        self.emotion_model = load_emotion_model()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Unable to open camera")
            exit()
        
        # Create UI components
        self.create_widgets()
        self.setup_probability_chart()
        # Start update loops
        self.update_interval = 50  # milliseconds
        self.update_camera()
        self.update_emoji()

    def create_widgets(self):
        # Configure grid column weights
        self.window.columnconfigure(0, weight=6)
        self.window.columnconfigure(1, weight=1)
        self.window.columnconfigure(2, weight=6)
        
        # Header styling
        self.header = Label(self.window, 
                       text="Real-Time Emotion Detection",
                       font=('Arial', 24),
                       bg='white',
                       fg='black')
        self.header.grid(row=0, column=0, columnspan=3, pady=20, sticky='ew')
        
        # Video feed container
        self.video_label = Label(self.window, 
                                bd=10,
                                bg='white')
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        # Emotion display
        self.emotion_label = Label(self.window, text="Neutral", 
                                font=('Arial', 32), bg='white', fg='white')
        self.emotion_label.grid(row=1, column=1, padx=(40,40), pady=20, 
                            sticky='nsew')

        # Probability chart
        self.setup_probability_chart()

        # Exit button
        self.exit_btn = Button(self.window, text="EXIT", command=self.close_app,
                            font=('Arial', 20), fg='red')
        self.exit_btn.grid(row=3, column=0, columnspan=3, pady=20, sticky='s')

        # Window size
        self.window.geometry("1500x900")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Emotion prediction
            emotion_idx = 4
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = np.expand_dims(np.expand_dims(roi, -1), 0) / 255.0
                preds = self.emotion_model.predict(roi, verbose=0)
                emotion_idx = np.argmax(preds)

                preds = self.emotion_model.predict(roi)
                probabilities = preds[0]
                self.update_chart(probabilities)

                # Draw bounding box
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0,255,0), 2)
                cv2.putText(frame, emotion_dict[emotion_idx], (x+20, y-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            # Display frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((800, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.emotion_label.configure(text=emotion_dict[emotion_idx])
        
        self.window.after(self.update_interval, self.update_camera)
    
    def setup_probability_chart(self):
        """Light-themed chart configuration"""
        self.figure = plt.Figure(figsize=(9,6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # Chart styling
        self.ax.set_facecolor('white')
        self.figure.patch.set_facecolor('white')
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black') 
        self.ax.spines['right'].set_color('black')
        self.ax.spines['left'].set_color('black')
        
        # Text colors
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.tick_params(axis='x', colors='black')
        self.ax.tick_params(axis='y', colors='black')

        # Bar chart styling
        self.bars = self.ax.barh(
            list(emotion_dict.values()),
            [0.1]*7,
            color='#1E90FF',
            height=0.6
        )
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.chart_widget = self.canvas.get_tk_widget()
        self.chart_widget.grid(row=1, column=2, rowspan=2, 
                            padx=(10, 30), pady=10, sticky='nsew')
        
        # Initial data
        self.emotion_names = list(emotion_dict.values())
        self.bars = self.ax.barh(self.emotion_names, [0]*7)
        self.ax.set_xlim(0, 1)
        self.ax.set_title('Prediction Probabilities')
    
    def update_chart(self, probabilities):
        """Update bar chart data"""
        for bar, prob in zip(self.bars, probabilities):
            bar.set_width(prob)
        self.figure.canvas.draw()

    def update_emoji(self):
        # Update emoji display
        current_emotion = self.emotion_label.cget("text")
        idx = list(emotion_dict.values()).index(current_emotion)
        try:
            emoji_img = Image.open(emoji_dist[idx]).resize((200, 200))
            emoji_tk = ImageTk.PhotoImage(emoji_img)
            self.emotion_label.emoji = emoji_tk
            self.emotion_label.configure(image=emoji_tk)
        except Exception as e:
            print(f"Emoji loading error: {str(e)}")
        self.window.after(300, self.update_emoji)

    def close_app(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()