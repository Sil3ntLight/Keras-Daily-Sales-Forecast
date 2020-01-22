#Import libraries
import datetime
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql

db_connection_str = 'mysql+pymysql://root:SQLPASSWORD@localhost/DBNAME'
db_connection = create_engine(db_connection_str)
# Execute the query

query = 'select * from `commerce_order` INNER JOIN `commerce_orderitem` ON `commerce_order`.`order_number` = `commerce_orderitem`.`order_id`;'

features_cols = ['upc_code', 'dayofweek', 'year', 'month', 'dayofyear', 'weekofyear', 'quarter' ]

#Define model
X_num_columns= len(features_cols)

model = Sequential()

model.add(Dense(300,
                activation='relu',
                input_dim = X_num_columns))

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

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model Created")

def chunks_to_df(gen):
    chunks = []
    for df in gen:
        chunks.append(df)
    return pd.concat(chunks).reset_index()


def runKeras(df):
    df['date_placed'] = df['date_placed'].astype('datetime64[D]')
    

    df = df.groupby(by=['date_placed', 'upc_code'],as_index=False).size().to_frame('count').reset_index()
    df['dayofweek'] = df['date_placed'].dt.dayofweek
    df['year'] = df['date_placed'].dt.year
    df['month'] = df['date_placed'].dt.month
    df['dayofyear'] = df['date_placed'].dt.dayofyear
    df['weekofyear'] = df['date_placed'].dt.weekofyear
    df['quarter'] = df['date_placed'].dt.quarter

    df.set_index('date_placed',inplace=True)
    dates = df.index
    print(df)
    Y = df.loc[:,'count']
    X = df.loc[:,features_cols]
    print(X)
    #Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)
    #Fit model to training data
    model.fit(X_train, y_train, epochs=5000,batch_size=1000)
    print("Training completed")




# Get data in batches
while True:
    df_chunks = pd.read_sql_query(query, con=db_connection, chunksize=10000)
    df = chunks_to_df(df_chunks)
    # We are done if there are no data
    if len(df) == 0:
        model.save("Sales_model.h5")
        break
    # Let's write to the file
    else:
        runKeras(df)


db_connection.close()
#Save trained model

print("Sales_model.h5 saved model to disk in ",os.getcwd())

#Predict known daily sales in order to check results
predictions = model.predict(X)
predictions_list = map(lambda x: x[0], predictions)
predictions_series = pd.Series(predictions_list,index=dates)
dates_series =  pd.Series(dates)

df_newDateslist = pd.date_range(start=datetime.datetime(2019, 12, 28), periods=300).tolist()
df_newDates = pd.Series(data=df_newDateslist).astype('datetime64[D]')
print(df_newDates)
print("Upcoming dates imported")

Predicted_sales = model.predict(df_newDates)

#Export predicted sales
new_dates_series=df_newDates
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)
new_predictions_series.to_csv("predicted_sales.csv")

