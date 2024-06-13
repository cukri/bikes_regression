import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dataset_cols = ['bike_count', 'hour', 'temp', 'humidity', 'wind','visibility', 'dew_pt_temp', 'radiation', 'rain'
    ,'snow','functional']
df= pd.read_csv('SeoulBikeData.csv').drop(['Date', 'Holiday','Seasons'], axis=1)
df.columns = dataset_cols
df['functional'] = (df['functional']=='Yes').astype(int)
df = df[df['hour']==12]
df = df.drop(['hour'], axis=1)
print(df)

def plotting():
    for label in df.columns[1:]:
        plt.scatter(df[label], df["bike_count"])
        plt.title(label)
        plt.ylabel("Bike Count at Noon")
        plt.xlabel(label)
        plt.show()

df = df.drop(['wind', 'visibility', 'functional'], axis=1)

#Split into Train,Val,test dataset
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def get_xy(dataframe, y_label,x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        x = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels)==1:
            x = dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            x = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1,1)
    data = {'x': x, 'y': y}

    return data, x, y

_, x_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels=["temp"])
_, x_val_temp, y_val_temp = get_xy(val, "bike_count", x_labels=["temp"])
_, x_test_temp, y_test_temp = get_xy(test, "bike_count", x_labels=["temp"])

temp_reg = LinearRegression()
temp_reg.fit(x_train_temp, y_train_temp)
s = temp_reg.score(x_test_temp, y_test_temp)
print(s)

#Multiple Linear Regression
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
_, x_train_all, y_train_all = get_xy(train, "bike_count", x_labels=df.columns[1:])
_, x_val_all, y_val_all = get_xy(val, "bike_count", x_labels=df.columns[1:])
_, x_test_all, y_test_all = get_xy(test, "bike_count", x_labels=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(x_train_all, y_train_all)
all_reg.score(x_test_all, y_test_all)

#Regression with Neural Network
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(x_train_temp.reshape(-1))

temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

def plot_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()

history = temp_nn_model.fit(x_train_temp.reshape(-1), y_train_temp,
                            verbose = 0,
                            epochs=1000,
                            validation_data=(x_val_temp, y_val_temp))

plot_history(history)