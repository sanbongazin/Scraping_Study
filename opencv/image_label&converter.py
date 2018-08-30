import random
import numpy as np

PATH_AND_LABEL = []

# パスとラベルが記載されたファイルから、それらのリストを作成する
with open("path_and_label.txt", mode='r') as file :
　　　　  for line in file :
　　　　　　　　   # 改行を除く
  　　　　   line = line.rstrip()
       # スペースで区切られたlineを、リストにする
       line_list = line.split()
       PATH_AND_LABEL.append(line_list)
　　　　　　　　　　　　　　# 同じジャンルのサムネイルが、かたまらないように、シャッフルする
       random.shuffle(PATH_AND_LABEL)


DATA_SET = []

for path_label in PATH_AND_LABEL :

    tmp_list = []

    # 画像を読み込み、サイズを変更する
    img = cv2.imread(path_label[0])
    img = cv2.resize(img, (28, 28))
    # (28, 28, 3)のN次元配列を一次元に、dtypeをfloat32に、０〜１の値に正規化する
    img = img.flatten().astype(np.float32)/255.0

    tmp_list.append(img)

    # 分類するクラス数の長さを持つ仮のN次元配列を作成する
    classes_array = np.zeros(3, dtype = 'float64')
    # ラベルの数字によって、リストを更新する
    classes_array[int(path_label[1])] = 1

    tmp_list.append(classes_array)

    DATA_SET.append(tmp_list)