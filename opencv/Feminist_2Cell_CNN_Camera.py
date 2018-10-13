
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


num_filters2 = 64

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, num_filters1, num_filters2],
                                           stddev = 0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                       strides=[1, 1, 1, 1], padding='SAME')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))

h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                         strides= [1, 2, 2, 1], padding = 'SAME')


# In[4]:


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


# In[5]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, '/Users/yamamotomasaomi/Documents/GitHub/Python_Study/opencv/learn_result_2cell_ver1.1-2000')
# ということで、演算はWindowsがやっても問題はない。しかし、画像の追加がやりづれlえので、USBが必要となりそう。
# saver.restore(sess, '/Users/yamamotomasaomi/Documents/GitHub/Python_Study/opencv/learn_result_2cell_2000')


# In[ ]:


import cv2 as cv
# from IPython.core.debugger import Pdb; Pdb().set_trace()

real_image, x_edit = [],[]

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "cascade.xml"
    cascade = cv.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv.namedWindow(ORG_WINDOW_NAME)
    cv.namedWindow(GAUSSIAN_WINDOW_NAME)
    
    i=0
    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))
        # 検出した顔に印を付ける
        for (x_, y, w, h) in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv.imwrite("cutted.jpg",cv.cvtColor(img,cv.COLOR_BGR2GRAY))
            img_read = cv.imread("cutted.jpg")
                    # フレーム表示
            cv.rectangle(img_gray, (x_, y), (x_+w, y+h), color, thickness = pen_w)
            img_cutter = img_read[y:y+h,x_:x_+w]
            cv.imwrite("famiimager/cutted"+str(i)+".jpg",img_cutter)
            img_read_cut = cv.imread("famiimager/cutted"+str(i)+".jpg")
            img_read_resized = cv.resize(img_read_cut,(28,28))
            real_image.append(img_read_resized.flatten().astype(np.float32)/255.0)
            x_tmp = np.reshape(real_image[i],(-1,2352))
            x_edit.append(x_tmp)
            p_val = sess.run(p, feed_dict={x:x_edit[i],keep_prob:1.0})
            print(np.argmax(p_val[0]))
            i+=1

        
        
        #今度は、これを画面上に表示できれば、リソースの測定は可能。ただし、正確性にかける可能性がある。
#         今度は、その画像を学習させることもできるので、これで正確性は増す可能性がある。
# というか、２層CNNでこの結果なんだから、正確性が実用的ではないかもしれない
        cv.imshow(GAUSSIAN_WINDOW_NAME, img_gray)
        # Escキーで終了
        key = cv.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv.destroyAllWindows()
    cap.release()


# In[ ]:


np.shape(x_image)

