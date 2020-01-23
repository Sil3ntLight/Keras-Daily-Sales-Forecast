#Import libraries
import datetime
import os
import numpy as np
import pandas as pd
from keras import metrics
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql
import sys

db_connection_str = 'mysql+pymysql://root:alpine5676@localhost/hist_sales'
db_connection = create_engine(db_connection_str)
# Execute the query

# change query depending on tables with sales data
query = 'select * from `commerce_order` INNER JOIN `commerce_orderitem` ON `commerce_order`.`order_number` = `commerce_orderitem`.`order_id`;'

# Change column names to columns that affect data. (features_cols[1:5] are generated later on based on column 'date_placed')
features_cols = ['upc_code', 'dayofweek', 'month', 'dayofyear', 'weekofyear', 'quarter' ]

#Define model
X_num_columns= len(features_cols)

if(len(sys.argv) < 2):
    model = Sequential()
    model.add(Dense(200,
                activation='relu',
                input_dim = X_num_columns))

    model.add(Dense(90,
                    activation='relu'))

    model.add(Dropout(0.4))

    model.add(Dense(30,
                    activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(7,
                    activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,
                    activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])
else:
    model = load_model(sys.argv[1])


print("Model Created")


def chunks_to_df(gen):
    chunks = []
    for df in gen:
        chunks.append(df)
    return pd.concat(chunks).reset_index()


def runKeras(df):
    # make sure date column is in datetime YYYY-MM-DD format
    df['date_placed'] = df['date_placed'].astype('datetime64[D]')

    # group by date, and then upc, and count how many of each upc was ordered for each date.
    df = df.groupby(by=['date_placed', 'upc_code'],as_index=False).size().to_frame('count').reset_index()

    # Fill other feature columns regarding date, determined by 'date_placed'
    df['dayofweek'] = df['date_placed'].dt.dayofweek
    df['month'] = df['date_placed'].dt.month
    df['dayofyear'] = df['date_placed'].dt.dayofyear
    df['weekofyear'] = df['date_placed'].dt.weekofyear
    df['quarter'] = df['date_placed'].dt.quarter

    # set 'date_placed' as index, also save to 'dates' variable
    df.set_index('date_placed',inplace=True)
    dates = df.index

    # split targets and features into their own dataframes (both with 'date_placed' index)
    Y = df.loc[:,'count']
    X = df.loc[:,features_cols]

    #Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)

    #Fit model to training data
    # Set epochs and batch_size here
    model.fit(X_train, y_train, epochs=1200,batch_size=50, use_multiprocessing=True)
    print("Batch completed")

    #Save trained model to runtime directory
    model.save("model.h5")
    print("model.h5 saved model to disk in ",os.getcwd())

    #Predict known daily sales in order to check results
    predictions = model.predict(X, use_multiprocessing=True)
    predictions_list = map(lambda x: x[0], predictions)
    predictions_series = pd.Series(predictions_list,index=dates)
    dates_series = pd.Series(dates)

    # get list of dates to predict for
    df_newDateslist = pd.date_range(start=datetime.datetime(2019, 12, 28), periods=180).tolist()
    df_newDates = pd.Series(data=df_newDateslist).astype('datetime64[D]')
    df_predict = pd.DataFrame()
    
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=100)
    print('test loss, test acc:', results)

    full_predictions_df = pd.DataFrame()
    
    # Predict sales for each day in generated list,
    # and do for each upc that shows up 
    for upc in df['upc_code'].unique():
        df_dates = pd.DataFrame()
        df_dates['date_placed'] = df_newDates
        df_dates['upc_code'] = upc
        df_predict_single = df_dates
        df_predict_single['dayofweek'] = df_predict_single['date_placed'].dt.dayofweek
        df_predict_single['month'] = df_predict_single['date_placed'].dt.month
        df_predict_single['dayofyear'] = df_predict_single['date_placed'].dt.dayofyear
        df_predict_single['weekofyear'] = df_predict_single['date_placed'].dt.weekofyear
        df_predict_single['quarter'] = df_predict_single['date_placed'].dt.quarter
        df_predict_single.set_index('date_placed',inplace=True)

        Predicted_sales = model.predict(df_predict_single,use_multiprocessing=True)
        new_dates_series=df_predict_single.index
        new_predictions_df = pd.DataFrame(Predicted_sales,index=new_dates_series)
        df_predict_single.reset_index(inplace=True)
        new_predictions_df.reset_index(inplace=True)
        df_predict = pd.merge(df_predict_single, new_predictions_df, left_on='date_placed', right_on='date_placed')
        full_predictions_df = pd.concat([full_predictions_df, df_predict]) 

    # Write predictions to csv file
    full_predictions_df.to_csv("predicted_sales.csv")

# Get data in batches from MySQL
while True:
    df_chunks = pd.read_sql_query(query, con=db_connection, chunksize=30000)
  
    for df in df_chunks:
    
        if len(df) == 0:
            break
        else:
            runKeras(df)

db_connection.close()




