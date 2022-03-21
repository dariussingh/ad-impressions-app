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

st.write(f"Given user traffic data from {data['Date'][0].strftime('%Y-%m-%d')} to {data['Date'][len(data)-1].strftime('%Y-%m-%d')}. Use number of steps for forecasting as a hyperparameter to finetune our model for user traffic forecasting for future dates.")
n_steps = st.slider('Number of steps for forecasting', 1, 12, 8)

# ARMA Model
model = ARIMA(data['User Traffic'], order=(n_steps,0,n_steps)) # (AR order, I order, MA order)
model = model.fit()

y = data['User Traffic'][n_steps:]
y_pred = model.predict(start=n_steps,end=len(data)-1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

st.write(f"The plot belows shows the user traffic forecasting model performance. It compares the True vs Predicted user traffic on the dataset from {data['Date'][0].strftime('%Y-%m-%d')} to {data['Date'][len(data)-1].strftime('%Y-%m-%d')}.")
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

st.write(f"Set the number of ad slots available to push advertisments.")
ad_slots = st.slider('Number of ad slots', 1, 10, 4)
st.write('Select number of ads required in dummy quote for n ads with some impression target by client.')
num_ads = st.slider('Number of ads quoted by client', 3,7,4)

st.write('Dummy data for Ad-wise number of impressions target. The dataset contains the unique identifier for each ad, its length, and the number of impression target that is given by client.')
ad_data = utils.create_random_ads(num_ads)
st.dataframe(ad_data)

# prerequsites for MC sim
ad_data['probability'] = utils.calculate_prob(ad_data['impression target'])
ad_data['cumulative_prob'] = utils.calculate_cumulative_probablity(ad_data['probability'])
ad_data['rand_interval'] = utils.create_random_interval(ad_data['cumulative_prob'])
lower_limit, upper_limit = utils.split_random_interval(ad_data['rand_interval'])
ad_data['lower_limit'] = lower_limit
ad_data['upper_limit'] = upper_limit

st.write('Select a date for which the ad impression modeling is to be run.')
date = st.date_input('Date', min_value=datetime(2022,1,1), value=datetime(2022,2,11))
date = date.strftime('%Y-%m-%d')

#--------------------------------------------------------------------------------

if st.button('Run Ad Impression Modeling'):
    st.header('Modeling Prediction')
    
    traffic, actual_exists = predict(date, data, n_steps=n_steps)
    opportunity_size = int(traffic*ad_slots)

    st.write(f"""
    On {date}:
    - User Traffic is between **{int(traffic-rmse)}** and **{int(traffic+rmse)}** users.
    - Opportunity Size is **{opportunity_size}**.
    - The Monte Carlo simulation of the ads pushed through according to their impression targets given the opportunity size.The horizontal dashed lines represent the impression target for the respective ads.
    """)
    rand_nums = utils.generate_random_num(int(opportunity_size/1000))
    sim = utils.run_simulation(rand_nums, ad_data)
    utils.plot_sim(sim, ad_data, num_ads, int(opportunity_size/1000))

    st.write(f"""
    ## Result Summary
    On {date} the expected user traffic is between **{int(traffic-rmse)}** and **{int(traffic+rmse)}** users. There are **{ad_slots}** available and as user traffic is **{int(traffic)}** (average of interval), we determine the opportunity size (= user traffic * number of ad slots) to be **{opportunity_size}**.

    We run a simulation to determine how to best push the ads to the user traffic given the ad slots, in order to complete the impressions target quote by the client. The above graph shows the result of the same. Where we see the each solid line shows how each ad should be pushed in order to fill the target ad quote by the client. The dashed lines show the target impressions thershold for each ad.
    """)

    st.write("""
    Below it can be seen the exact order in which to push each ad onto every 1000 opportunities in order to fulfill the impressions target set by the client. Where each row represents 1000 opportunities (i.e. one ad slot for 1000 users) and the id represents which add to push for the opportunity.
    """)
    st.dataframe(sim['ids'])