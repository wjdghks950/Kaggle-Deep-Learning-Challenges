
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os
import tensorflow as tt
from random import shuffle
from tqdm import tqdm

TRAIN_DIR ='C:/Users/JeonghwanKim/Desktop/train'
TEST_DIR ='C:/Users/JeonghwanKim/Desktop/test'
IMG_SIZE = 100
LR = 1e-4

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic-video')


# In[3]:


def label_img(img):
    #dog.9333.jpg
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1, 0] #if model_out == 0, then cat  (model_out gives index)
    elif word_label == 'dog': return [0, 1] #if model_out == 1, then dog  (model_out gives index)


# In[4]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        if height > width:
            diff = height - width
            if diff % 2 == 0:
                sub = int(diff/2)
                img = np.lib.pad(img, ((0,0), (sub, sub)), 'constant')
            else:
                sub = int(diff/2)
                img = np.lib.pad(img, ((0,0), (sub, sub+1)), 'constant')
        elif height < width:
            diff = width - height
            if diff % 2 == 0:
                sub = int(diff/2)
                img = np.lib.pad(img, ((sub, sub), (0,0)), 'constant')
            else:
                sub = int(diff/2)
                img = np.lib.pad(img, ((sub, sub+1), (0,0)), 'constant')
        
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        
        batch_mean, batch_var =tt.nn.moments(img, shape = [0, 1, 2]) #Calculate mean and variance of resized images
        
        var_epsilon = 1e-3 #providing var_epsilon to avoid dividing by 0
        
        img = tt.nn.batch_normalization(img, batch_mean, batch_var, var_epsilon) #input image normalization

        training_data.append([np.array(img), np.array(label)])
                                 
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[5]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0] # i.e. [1039.jpg] is the file format
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data


# In[6]:


train_data = np.load('train_data.npy')
#if you already have train data:
#train_data = create_train_data()


# In[7]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#100x100x1

convnet = conv_2d(convnet, 50, 3, activation='relu') #3x3x50 --> 98x98x50
convnet = max_pool_2d(convnet, 2) #2x2 -->49x49x50

convnet = conv_2d(convnet, 50, 3, activation='relu') #3x3x50 --> 47x47x50
convnet = max_pool_2d(convnet, 2) #2x2 --> 23x23x50

convnet = conv_2d(convnet, 50, 3, activation='leaky_relu') #3x3x50 --> 21x21x50 
convnet = max_pool_2d(convnet, 2) #2x2 --> 10x10x50

convnet = conv_2d(convnet, 50, 3, activation='leaky_relu') #3x3x50 --> 8x8x50
#convnet = max_pool_2d(convnet, 2) #2x2 --> 4x4x50

convnet = fully_connected(convnet, 3200, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 400, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 200, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='rmsprop', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[8]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[11]:


train = train_data[:-500]
test = train_data[-500:]


# In[12]:


X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #Pixel data
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #Pixel data
test_Y = [i[1] for i in test]


# In[13]:


model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_Y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME) 


# In[14]:


model.save(MODEL_NAME)


# In[11]:


import matplotlib.pyplot as plt
from math import ceil

test_data = process_test_data()

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    #cat: [1:0]
    #dog: [0:1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label = 'Dog'
    else: str_label = 'Cat'
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()


# In[12]:


with open('submission-file.csv', 'w') as f:
    f.write('id,label\n')


# In[ ]:


with open('submission-file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        mod_out = round(model_out[1])
        f.write('{},{}\n'.format(img_num, mod_out))
        #f.write('{},{}\n'.formate(img_num, model_out[1])) ---- for Kaggle [Dogs vs. Cats Redux. Kernel edition competition]


# In[ ]:




