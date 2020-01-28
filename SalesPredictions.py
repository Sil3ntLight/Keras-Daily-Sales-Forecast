#!/usr/bin/env python
# coding: utf-8

# In[139]:


load = True
evaluate = True
doTrain = True
model_file = "model"


# In[140]:



EVALUATION_INTERVAL = 250 
EPOCHS = 10
BATCH_SIZE = 32
BUFFER_SIZE = 10000
future_target = 100
STEP = 20


# In[141]:



import datetime
from datetime import datetime as dt
import os
import os.path
from os import path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql
import shutil


# In[142]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[143]:


def date_parser(date_str):
    return dt.strptime(date_str, "%Y-%m-%d").strftime("%s")


# In[144]:


if(path.exists('./logs/')):
    shutil.rmtree('./logs/')


# In[145]:


db_connection_str = 'mysql+pymysql://root:alpine5676@10.0.0.20:32768/kuhl_prod'
db_connection = create_engine(db_connection_str)


# In[146]:


query = 'select * from `commerce_order` INNER JOIN `commerce_orderitem` ON `commerce_order`.`order_number` = `commerce_orderitem`.`order_id`;'


# In[147]:


logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1,profile_batch = 1000000)


# In[148]:


def create_time_steps(length):
  return list(range(-length, 0))


# In[149]:


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    
    labels.append(target[i:i+target_size])


  return np.array(data), np.array(labels)


# In[150]:


def chunks_to_df(gen):
    chunks = []
    for df in gen:
        chunks.append(df)
    return pd.concat(chunks).reset_index()


# In[151]:


def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 5]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


# In[152]:


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=200):
  dataframe = dataframe.copy()
  labels = dataframe.pop('count')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# In[153]:


# Change column names to columns that affect data. (some of features_cols are generated later on based on column 'date_placed')
features_cols = ['upc_code', 'style_no', 'color_code', 'waist', 'size', 'dayofweek', 'month', 'dayofyear', 'weekofyear', 'quarter', 'count']
num_cols = ['upc_code', 'style_no', 'dayofweek', 'month', 'dayofyear', 'weekofyear', 'quarter', 'count']
#Define model
X_num_columns= len(features_cols)


# In[154]:


df = pd.read_sql_query(query, con=db_connection)


# In[ ]:



feature_columns = []
tf.random.set_seed(13)
# make sure date column is in datetime YYYY-MM-DD format
df['date_placed'] = df['date_placed'].astype(str) + ' 00:00:00.00' 
df['date_placed'] = pd.to_datetime(df['date_placed'])

df_old = df
# group by date, and then upc, and count how many of each upc was ordered for each date.
df = df.groupby(by=['date_placed', 'upc_code'],as_index=False).size().to_frame('count').reset_index()
# Fill other feature columns regarding date, determined by 'date_placed'
df['year'] = pd.to_numeric(df['date_placed'].dt.year)
df['dayofweek'] = pd.to_numeric(df['date_placed'].dt.dayofweek)
df['month'] = pd.to_numeric(df['date_placed'].dt.month)
df['dayofyear'] = pd.to_numeric(df['date_placed'].dt.dayofyear)
df['weekofyear'] = pd.to_numeric(df['date_placed'].dt.weekofyear)
df['quarter'] = pd.to_numeric(df['date_placed'].dt.quarter)
df['upc_code'] = pd.to_numeric(df['upc_code']) 
df['count'] = pd.to_numeric(df['count'])
df_old['upc_code'] = pd.to_numeric(df_old['upc_code']) 
df = df.merge(df_old[['upc_code','waist','style_no','color_code','size']], how='inner', on='upc_code')

df = df.drop('date_placed', axis=1)


waist = feature_column.categorical_column_with_vocabulary_list('waist', df['waist'].unique().tolist())
waist_embedding = feature_column.embedding_column(waist, dimension=8)
## color_code
color_code = feature_column.categorical_column_with_vocabulary_list('color_code', df['color_code'].unique().tolist())
color_code_embedding = feature_column.embedding_column(color_code, dimension=8)
## size
size = feature_column.categorical_column_with_vocabulary_list('size', df['size'].unique().tolist())
size_embedding = feature_column.embedding_column(size, dimension=8)


feature_columns.append(waist_embedding)
feature_columns.append(color_code_embedding)    
feature_columns.append(size_embedding)  
# add to keras model
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# create seperated datasets
train, test = train_test_split(df, test_size=0.2)
print(len(train), 'train examples')
print(len(test), 'test examples')

train_ds = df_to_dataset(train, shuffle=True, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test, shuffle=False, batch_size=BATCH_SIZE)

if(load):
    model = load_model('model')

if (doTrain):
    model = Sequential()
    model.add(feature_layer)
    model.add(Dense(300,
                    activation='relu'))
    model.add(Dense(90,
                        activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30,
                        activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(7,
                        activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,
                        activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    print("Model Created")
    model.fit(train_ds, epochs=EPOCHS,callbacks=[tensorboard_callback])
    model.save('model')


        


# In[ ]:


model = load_model('model')

if (evaluate):

    print('\n# Evaluate on test data')
    results = model.evaluate(test_ds)
    print('test loss, test acc:', results)


# In[ ]:


def predict_per_upc(predict_df, upc):
    predict_df = predict_df.reset_index(drop=True)
    
    predict_ds = df_to_dataset(predict_df, shuffle=False, batch_size=BATCH_SIZE)
    Predicted_sales = model.predict(predict_ds)
    

    new_predictions_series = pd.DataFrame(Predicted_sales, columns=['count'])
    new_predictions_series = new_predictions_series.reset_index(drop=True)
    new_predictions_df = predict_df.reset_index(drop=True)
    new_predictions_df['count'] = new_predictions_series['count']

    new_predictions_df.to_csv("./predictions/" + str(upc) + "_predicted-sales.csv")
    print(str(upc) + " Done! Exported to ./predictions/"+str(upc)+"_predicted-sales.csv")


# In[ ]:


# create dataframe to predict for
predict_df_upc = pd.DataFrame(columns=features_cols)

predict_df_upc['upc_code'] = pd.DataFrame(df['upc_code'].unique(), columns=['upc_code'])['upc_code']


predict_df_upc = predict_df_upc.merge(df[['upc_code','waist','style_no','color_code','size']], how='inner', on='upc_code')

group = predict_df_upc.groupby('upc_code')

for upc, data in group:
    predict_df = pd.DataFrame(columns=features_cols)
    predict_df['date_placed'] = ""
    predict_df['count'] = ""
    predict_df_dates = pd.date_range(start="01/28/2020", periods=future_target)
    for date in predict_df_dates.values:
        date_upc_df = data
        date_upc_df['date_placed'] = date
        predict_df = predict_df.append(date_upc_df, sort=True)
    # Fill other predict columns regarding date, determined by 'date_placed'
    predict_df['date_placed'] = pd.to_datetime(predict_df['date_placed'])
    predict_df['year'] = pd.to_numeric(predict_df['date_placed'].dt.year)
    predict_df['dayofweek'] = pd.to_numeric(predict_df['date_placed'].dt.dayofweek)
    predict_df['month'] = pd.to_numeric(predict_df['date_placed'].dt.month)
    predict_df['dayofyear'] = pd.to_numeric(predict_df['date_placed'].dt.dayofyear)
    predict_df['weekofyear'] = pd.to_numeric(predict_df['date_placed'].dt.weekofyear)
    predict_df['quarter'] = pd.to_numeric(predict_df['date_placed'].dt.quarter)
    predict_df['waist'] = predict_df['waist_y']
    predict_df['style_no'] = predict_df['style_no_y']
    predict_df['color_code'] = predict_df['color_code_y']
    predict_df['size'] = predict_df['size_y']

    predict_df.drop('style_no_x',inplace=True, axis=1)
    predict_df.drop('style_no_y',inplace=True, axis=1)
    predict_df.drop('color_code_x',inplace=True, axis=1)
    predict_df.drop('color_code_y',inplace=True, axis=1)
    predict_df.drop('waist_x',inplace=True, axis=1)
    predict_df.drop('waist_y',inplace=True, axis=1)
    predict_df.drop('size_x',inplace=True, axis=1)
    predict_df.drop('size_y',inplace=True, axis=1)
    predict_df.drop('date_placed',inplace=True, axis=1)
    predict_df['waist'] = predict_df['waist'].fillna("_")
    predict_df['color_code'] = predict_df['color_code'].fillna("_")
    predict_df['size'] = predict_df['size'].fillna("_")
    predict_df['style_no'] = predict_df['style_no'].fillna("_")
    
    predict_df = predict_df.drop_duplicates(subset=['year','dayofyear'])

    predict_per_upc(predict_df, upc)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[ ]:




