import glob
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical

def is_not_blank(s):
    return bool(s and s.strip())

sleep_stage = {
      'Wake': 0
    , 'N1': 1
    , 'N2': 2
    , 'N3': 3
    , 'REM': 4
}

data_path = '/DATA/*/'
csv_path = '/DATA/trainset-for_user.csv'

sleep_data_file = tf.keras.utils.get_file("trainset-for_user.csv", 'file://'+csv_path)
df = pd.read_csv(sleep_data_file, encoding='utf-8', names = ['folder', 'img_path', 'label'])

df['img_path'] = df['img_path'].str.strip()
df['folder'] = df['folder'].str.strip()
df['label'] = df['label'].str.strip()

folder_list = glob.glob(data_path)

def trainGenerator():
    for folder in folder_list:
        print(folder)
        labels = []
        features = []
        image_list = glob.glob(folder+'*')
        image_list.sort()

        for image_path in image_list:
            image_path = image_path.strip()
            series = df.loc[df['img_path'] == image_path[23:]]
            if series.size == 0 :
                continue
            try:
                #labeling
                labels.append(sleep_stage[series['label'].item()])
                #featureing
                img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img.reshape(-1,480)
                img = np.pad(img, ((5,5),(0,0)), constant_values=(0))
                img /= 255.0
                img = img.reshape(1,280,480,1)
                features.append(img)
            except Exception:
                print('Data preprocessing Error>>>>>>>>>>>>>')

        features = np.array(features).reshape(-1, 280, 480, 1)
        labels = np.array(labels).reshape(-1, 1)

        yield features, labels

def valGenerator():
    for folder in folder_list:
        print(folder)
        labels = []
        features = []
        image_list = glob.glob(folder+'*')
        image_list.sort()
        i = 0
        for image_path in image_list:
            image_path = image_path.strip()
            series = df.loc[df['img_path'] == image_path[23:]]
            if series.size == 0 :
                continue
            try:
                #labeling
                labels.append(sleep_stage[series['label'].item()])
                #featureing
                img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img.reshape(-1,480)
                img = np.pad(img, ((5,5),(0,0)), constant_values=(0))
                img /= 255.0
                img = img.reshape(1,280,480,1)
                features.append(img)
                i+=1
                if i > 128 :
                    break
            except Exception:
                print('Data preprocessing Error>>>>>>>>>>>>>')

        features = np.array(features).reshape(-1, 280, 480,1)
        labels = np.array(labels).reshape(-1, 1)
        yield features, labels
