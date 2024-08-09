#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the dataset
data_path = "C:\\Users\\NASIR KHAN\\Downloads\\CELEB DF V2"
real_videos_path = os.path.join(data_path, "Celeb-real")
fake_videos_path = os.path.join(data_path, "fake")
youtube_real_videos_path = os.path.join(data_path, "YouTube-real")
test_list_path = os.path.join(data_path, "List_of_testing_videos.txt")


# In[ ]:





# In[3]:


# pip install opencv-python


# In[4]:


# pip install opencv-python-headless


# In[5]:


# pip install opencv-contrib-python


# In[6]:


import cv2
print(cv2.__version__)


# In[7]:


with open(test_list_path, 'r') as file:
    test_videos = file.read().splitlines()


# In[8]:


def extract_frames(video_path, label, frame_rate=5):
    frames = []
    labels = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / frame_rate))
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))  # Resizing for consistency
            frames.append(frame)
            labels.append(label)
    cap.release()
    return frames, labels






# In[9]:


real_frames, real_labels = [], []
fake_frames, fake_labels = [], []


# In[10]:


# Extract frames from real videos
for video in os.listdir(real_videos_path):
    video_path = os.path.join(real_videos_path, video)
    frames, labels = extract_frames(video_path, 0)  # 0 for real
    real_frames.extend(frames)
    real_labels.extend(labels)


# In[11]:


# Extract frames from fake videos
for video in os.listdir(fake_videos_path):
    video_path = os.path.join(fake_videos_path, video)
    frames, labels = extract_frames(video_path, 1)  # 1 for fake
    fake_frames.extend(frames)
    fake_labels.extend(labels)


# In[12]:


from sklearn.model_selection import train_test_split

all_frames =np.array(real_frames + fake_frames)
all_labels = np.array(real_labels + fake_labels)

X_train, X_val, y_train, y_val = train_test_split(all_frames, all_labels, test_size=0.2, random_state=42)


# In[ ]:


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


real_frames


# In[ ]:




