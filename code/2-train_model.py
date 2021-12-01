import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from utils import model
from utils import processing_functions
from sklearn.preprocessing import OneHotEncoder
import joblib

# get pitch dataframe
df = processing_functions.get_pitch_table()
df['label'] = df['pitch'].isin(['CU','SL','CH','KC','FC','EP'])*1
df['month'] = pd.to_datetime(df['date'],format='%Y-%m-%d').dt.strftime('%Y-%m')
df = df.loc[df['date']>'2017-01-01']
y = df['label']

# create one-hot-encoded metadata dataset
enc = OneHotEncoder(sparse=False)
X_meta_data = enc.fit_transform(df[['batter','month']])
with open('../trained_models/pitch_model/metadata_processor.joblib', 'wb') as f:
    joblib.dump(enc, f)

# create image dataset
X_image_data = processing_functions.get_images(df['pitch_id'].values)



# create train and test datasets
X_image_train, X_image_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(X_image_data,X_meta_data, y,test_size=0.2)



# fit model
input_shape=(128, 108, 1)
batch_size = 256
num_epoch = 100

pitch_model = model.PitchModel(image_shape = input_shape)
pitch_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',tf.keras.metrics.AUC()])
early_stopping = model.EarlyStoppingWithThreshold(monitor='val_accuracy', patience=10,threshold=0.6)


model_log = pitch_model.fit(x= [X_image_train,X_meta_train], y= y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          validation_data=([X_image_test,X_meta_test], y_test),
          callbacks=[early_stopping],
          verbose=1)


print(pitch_model.summary())
pitch_model.save(os.path.join('../trained_models/pitch_model/model'))