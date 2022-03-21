import utils
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import streamlit as st

#--------------------------------------------------------------------------------
# set config
st.set_page_config(layout='wide')
#--------------------------------------------------------------------------------
st.title('Advertisement Impression Modeling')

st.header('Input parameters')

# time series forecasting
data = pd.read_excel('Daily Users.xlsx').copy()
data['Date'] = pd.to_datetime(data['Date'], format="%b %d '%y")
data['Day'] = data['Date'].apply(lambda x: x.strftime('%A'))
data['Month'] = data['Date'].apply(lambda x: x.strftime('%B'))

n_steps = st.slider('Number of steps for forecasting', 1, 14, 8)

# ARMA Model
model = ARIMA(data['User Traffic'], order=(n_steps,0,n_steps)) # (AR order, I order, MA order)
model = model.fit()

y = data['User Traffic'][n_steps:]
y_pred = model.predict(start=n_steps,end=len(data)-1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

st.write('The plot belows shows the user traffic forecasting model performance')
fig = plt.figure(figsize=(16,5), dpi=100)
plt.plot(data['Date'][n_steps:], y, label='True')
plt.plot(data['Date'][n_steps:], y_pred, label='Predicted')
plt.title('True vs Predicted User Traffic')
plt.xlabel('Date')
plt.ylabel('User Traffic')
plt.legend()
st.pyplot(fig)

def predict(date, data, n_steps):
    date = datetime.strptime(date, '%Y-%m-%d')
    last_date = str(data['Date'].values[-1]).split('T')[0]
    last_date = datetime.strptime(last_date, '%Y-%m-%d')
    min_date = str(data['Date'].values[0]).split('T')[0]
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    
    if min_date<=date<=last_date:
        actual_exists = True
        traffic = data[data['Date']==date]
        idx = traffic.index[0]
        traffic = model.predict(start=idx,end=idx+1)
        traffic = list(traffic)[-1]
    elif date>last_date:
        actual_exists = False
        delta = date-last_date
        delta = delta.days
        traffic = model.predict(start=len(data)-1,end=len(data)-1+delta)
        traffic = list(traffic)[-1]
    else:
        return 'Error', False
    
    return traffic, actual_exists

ad_slots = st.slider('Number of ad slots', 1, 10, 4)
date = st.date_input('Date', min_value=datetime(2022,1,1), value=datetime(2022,2,11))
date = date.strftime('%Y-%m-%d')

st.write('Ad-wise number of impressions target')
ad_data = utils.create_random_ads(5)
st.dataframe(ad_data)

# prerequsites for MC sim
ad_data['probability'] = utils.calculate_prob(ad_data['impression target'])
ad_data['cumulative_prob'] = utils.calculate_cumulative_probablity(ad_data['probability'])
ad_data['rand_interval'] = utils.create_random_interval(ad_data['cumulative_prob'])
lower_limit, upper_limit = utils.split_random_interval(ad_data['rand_interval'])
ad_data['lower_limit'] = lower_limit
ad_data['upper_limit'] = upper_limit


if st.button('Run Ad Impression Modeling'):
    st.header('Modeling Prediction')
    
    traffic, actual_exists = predict(date, data, n_steps=n_steps)
    opportunity_size = int(traffic*ad_slots)

    st.write(f"""
    On {date}:
    - User Traffic is between **{int(traffic-rmse)}** and **{int(traffic+rmse)}** users.
    - Opportunity Size is **{opportunity_size}**.
    """)
    rand_nums = utils.generate_random_num(int(opportunity_size/100))
    sim = utils.run_simulation(rand_nums, ad_data)
    utils.plot_sim(sim, ad_slots, int(opportunity_size/100))
