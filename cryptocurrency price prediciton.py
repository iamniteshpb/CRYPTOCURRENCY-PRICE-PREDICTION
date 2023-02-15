import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datatime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import sequential

crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)

#prepare data
scaler = MinMaxscaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

predictions_days = 60

x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x-train), np.array(y_train)
x_train = np.reshape(x-train, (x_train.shape[0], x_train.shape[1], 1))

#create Neural Network

model = Sequential()

model.add(LSTM(units=50, retur_sequences=True,) input_shape=(x_train.shape[1],1))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.Dense(units=1)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Testing The model

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
actual_prices = test_data['close'].values

total_dataset = pd.concat((data['close'], test_data['close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test,.shape[0], x_test.shape[1, 1]))

prediction_prices = model.predict(x_test)
prediction_prices = scalar.inverse_transform(prediction_prices)

plt.plot(actual_prices, color ='black', label='Actual Prices')
plt.plot(prediction_prices, color ='green', label='predicted Prices')
plt.title(f'[crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

#predict Next Day

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_dat = np.array(real_data)
real_dat = np.reshape(x_test.shape[0], x_test.shape[1], 1)

#create Neural Network

model + Sequential()






__________________
import pickle
from flask import Flask, request
import pandas as pd
import numpy as np


with open('C:/Users/NITESH/Downloads/model.pkl','rb') as model_svm_pickle:
    model_svm = pickle.load(model_svm_pickle)
    

ml_api = Flask(__name__)

@ml_api.route('/')
def welcome():
    return "Welcome All"

@ml_api.route('/predict_svc', methods=['GET'])
def predict_svc():
    title = request.args.get('title')
    author = request.args.get('author')
    price = request.args.get('price')
    pages = request.args.get('pages')
    avg_reviews = request.args.get('avg_reviews')
    n_reviews = request.args.get('n_reviews')
    star5 = request.args.get('star5')
    star4 = request.args.get('star4')
    star3 = request.args.get('star3')
    star2 = request.args.get('star2')
    star1 = request.args.get('star1')
    dimensions = request.args.get('dimensions')
    weight = request.args.get('weight')
    language = request.args.get('language')
    publisher = request.args.get('publisher')
    ISBN_13 = request.args.get('ISBN_13')
    link = request.args.get('link')
    complete_link = request.args.get('complete_link')
    input_data = np.array([[title,author,price,pages,avg_reviews,n_reviews,
       star5,star4,star3,star2,star1,dimensions,weight,
       language,publisher,ISBN_13,link,complete_link]])
    prediction = model_svm.predict(input_data)
    return "prediction is " + str(prediction)

@ml_api.route('/predict_sv_file', methods=['POST'])
def predict_sv_file():
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model_svm.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    ml_api.run(host = '0.0.0.0', port=5000)
