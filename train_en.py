import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress logs
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.metrics import classification_report

# Data paths and generator configurations
train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(
    rescale=1./255,                                      # Normalization
    rotation_range=30,                                   # Random rotation ±30 degrees
    width_shift_range=0.2,                               # Horizontal shifting
    zoom_range=0.2,                                      # Random zoom
    horizontal_flip=True                                 # Horizontal flipping
)

val_datagen = ImageDataGenerator(rescale=1./255)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

early_stop = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=10,          # Stop after 10 epochs without improvement
    restore_best_weights=True  # Restore best model weights
)

def visualize_augmentation(generator):
    plt.figure(figsize=(15, 8))
    for i in range(10):
        x, y = next(generator)
        plt.subplot(2,5,i+1)
        plt.imshow(x[0].squeeze(), cmap='gray')
        plt.title(emotion_dict[y[0].argmax()])
        plt.axis('off')
    plt.suptitle('Augmented Training Samples')
    plt.savefig('data_augmentation.png')
    
visualize_augmentation(train_generator)

# Model definition
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

def load_emotion_model(model_weights_path):
    # Define model architecture (must match training architecture)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    # Load weights
    model.load_weights(model_weights_path)
    return model

# Compile and train
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# Training history callback
class TrainingHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()  # Explicitly call parent initializer
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}  # Handle empty logs
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

history = TrainingHistory()

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=7178 // 64,
    callbacks=[history, early_stop]
)

# Debug output
print(f"Training loss records: {len(history.losses)}")
print(f"Validation accuracy records: {len(history.val_acc)}")

# Visualization function with error handling
def plot_training_history(history):
    try:
        plt.figure(figsize=(12, 6))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(history.losses, label='Training Loss')
        plt.plot(history.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(history.acc, label='Training Accuracy')
        plt.plot(history.val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
    except AttributeError as e:
        print(f"Plotting failed: {str(e)}")
        print("Please check callback data collection")

plot_training_history(history)

# Save model
emotion_model.save_weights('emotion_model.h5')

# Confusion matrix
def plot_confusion_matrix():
    # Get validation data
    val_images = []
    val_labels = []
    for i in range(len(validation_generator)):
        x, y = validation_generator[i]
        val_images.append(x)
        val_labels.append(y.argmax(axis=1))
    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)
    
    # Predictions
    preds = emotion_model.predict(val_images)
    pred_labels = preds.argmax(axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, pred_labels)
    # Classification report
    report = classification_report(val_labels, pred_labels, 
                                  target_names=emotion_dict.values(),
                                  output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Visualization layout
    plt.figure(figsize=(18,6))
    
    # Confusion matrix
    plt.subplot(131)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotion_dict.values(),
               yticklabels=emotion_dict.values())
    plt.title('Confusion Matrix')
    
    # Normalized matrix
    plt.subplot(132)
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_cm, annot=True, fmt='.2f', cmap='Oranges',
               xticklabels=emotion_dict.values(),
               yticklabels=emotion_dict.values())
    plt.title('Normalized Matrix')
    
    # Metrics heatmap
    plt.subplot(133)
    metrics_map = report_df[['precision', 'recall', 'f1-score']].iloc[:-3]
    sns.heatmap(metrics_map, annot=True, fmt='.2f', cmap='Greens')
    plt.title('Classification Metrics')
    
    plt.tight_layout()
    plt.savefig('enhanced_confusion_matrix.png')
    
plot_confusion_matrix()

# Camera detection code
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    bounding_box = cv2.CascadeClassifier(casc_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()