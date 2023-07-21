import numpy as np
from keras.models import load_model
from util import csv_to_dataset, history_points

model = load_model('technical_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('AAPL_intraday.csv')

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

buys = []
sells = []
thresh = 0.1

start = 0
end = -1

x = -1



import numpy as np
from keras.models import load_model
from util import csv_to_dataset, history_points
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update_plot(frame):
    global x, start, end, buys, sells

    x += 1

    if x >= len(ohlcv_test):
        ani.event_source.stop()
        return

    # Update the plot data with new predictions
    y_pred = model.predict([ohlcv_test[x:x+1], tech_ind_test[x:x+1]])
    y_pred = y_normaliser.inverse_transform(y_pred)
    
    plt.cla()  # Clear the current plot
    plt.plot(unscaled_y_test[start:end], label='True Values', color='blue')
    plt.plot(y_test_predicted[start:end], label='Predicted Values', color='orange')
    
    if y_pred > unscaled_y_test[x] + thresh:
        buys.append((x, unscaled_y_test[x]))
    elif y_pred < unscaled_y_test[x] - thresh:
        sells.append((x, unscaled_y_test[x]))

    # Plot buy/sell markers
    if buys:
        buy_x, buy_y = zip(*buys)
        plt.scatter(buy_x, buy_y, marker='^', color='g', label='Buy')
    if sells:
        sell_x, sell_y = zip(*sells)
        plt.scatter(sell_x, sell_y, marker='v', color='r', label='Sell')
    
    plt.legend()
    plt.title('Real-time Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')

API_KEY = 'FLBKFNSA95C7STE0'

def get_live_stock_price(symbol):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    return float(data['Global Quote']['05. price'])

# Replace 'AAPL' with the stock symbol you want to track
stock_symbol = 'AAPL'

# Create a figure and start the animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, frames=len(ohlcv_test), interval=100)  # Adjust the interval as needed
plt.show()
