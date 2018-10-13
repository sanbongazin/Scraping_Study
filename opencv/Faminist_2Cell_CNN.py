
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cross_validation import train_test_split
from PIL import Image
import cv2 as cv
import os
import numpy.random as rd

np.random.seed(19960604)

class NotFaminist:
    def __init__(self):
        images, labels = [], []

        for i, letter in enumerate(['0', '1']):
            directory = '/Users/yamamotomasaomi/Documents/Github/Python_Study/opencv/notFaminist/%s/' % letter
            files = os.listdir(directory)
            label = np.array([0]*3)
            label[i] = 1
            i = 0
            for file in files:
                try:
                    img = cv.imread(directory+file)
                    img = cv.resize(img,(28,28))
                    images.append(img.flatten().astype(np.float32)/255.0)
                except:
                    print ("Skip a corrupted file: " + file)
                    continue
                labels.append(label)
            train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=0)
        
        class train:
            def __init__(self):
                self.images = []
                self.labels = []
                self.batch_counter = 0
                
            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    rd.shuffle(images)
                    rd.shuffle(labels)
                    batch_images = self.images[self.batch_counter:]
                    batch_labels = self.labels[self.batch_counter:]
                    left = num - len(batch_labels)
                    batch_images.extend(self.images[:left])
                    batch_labels.extend(self.labels[:left])
                    self.batch_counter = left
                else:
                    batch_images = self.images[self.batch_counter:self.batch_counter+num]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter+num]                  
                    self.batch_counter += num
                    
                return (batch_images, batch_labels)
                    
        class test:
            def __init__(self):
                self.images = []
                self.labels = []
                
        self.train = train()
        self.test = test()
                
        self.train.images = train_images
        self.train.labels = train_labels
        self.test.images = test_images
        self.test.labels = test_labels


# In[2]:


mnist = NotFaminist()


# In[3]:


#  必要な素材の収集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from __future__ import division
import sys, os, pickle
import numpy.random as rd

from scipy.misc import imread

import matplotlib.pyplot as plt

# ランダムシードの作成
# np.random.seed(20160703)
# tf.set_random_seed(20160703)


# その他コンフィグ（いつもの）
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('matplotlib', 'notebook')

#画像取り込みデータのオープン
# !ls session_MNIST_Image*


# In[4]:


# 畳みこみフィルターの設定
num_filters1 = 32

# プレースホルダーの設定
x = tf.placeholder(tf.float32,[None, 28*28*3])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1],
                                         stddev = 0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                      strides = [1, 1, 1, 1], padding = 'SAME')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2 , 2 , 1],
                         strides = [1, 2, 2, 1], padding = 'SAME')


# In[5]:


num_filters2 = 64

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, num_filters1, num_filters2],
                                           stddev = 0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                       strides=[1, 1, 1, 1], padding='SAME')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))

h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                         strides= [1, 2, 2, 1], padding = 'SAME')


# In[6]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 3*7*7*num_filters2])

num_units1 = 3*7*7*num_filters2
num_units2 = 510

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 3]))
b0 = tf.Variable(tf.zeros([3]))
p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)


# In[7]:


t = tf.placeholder(tf.float32, [None, 3])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# In[13]:


i=0
time = []
loss_val , acc_val = [], []
loss_vals, acc_vals = [], []

for _ in range (2000):
    i+=1
    batch_xs, batch_ts = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch_xs, t: batch_ts, keep_prob:1.0})
    time.append(i)
#     loss_val.append(loss_vals)
#     acc_val.append(acc_vals)
    
    if i % 500 == 0:
        for c in range(4):
            start = len(mnist.test.images) // 4 * c
            end = len(mnist.test.images) // 4 * (c+1)
            loss_vals, acc_vals = sess.run([loss,accuracy],feed_dict={x:mnist.test.images[start:end],t:mnist.test.labels[start:end],keep_prob:1.0})
        loss_val_sum = np.sum(loss_vals)
        acc_val_sum = np.mean(acc_vals)
        print('Step: %d, Loss:%f, Accuracy: %f' 
                % (i, loss_val_sum, acc_val_sum))
saver.save(sess, '/Users/yamamotomasaomi/Documents/GitHub/Python_Study/opencv/learn_result_2cell', global_step=i)



