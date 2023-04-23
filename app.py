import streamlit as st 
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import joblib

#Load the model
model = joblib.load(r'c:/Users/Nii.Sowah/Desktop/DS/streamlitTutorial/hotel_booking_model.joblib')

#Now we are going to start creating our app

cols=['deposit_type', 'lead_time', 'adr', 'arrival_date_day_of_month',
         'arrival_date_week_number', 'total_of_special_requests', 'agent',
         'market_segment', 'previous_cancellations', 'stays_in_week_nights']

st.title("Did Customer Cancel ? :face_with_rolling_eyes:")

st.write('Please spare a minute or two to answer the Questions below, this will help us make our prediction')

deposit_type=st.selectbox("What type of deposite was made",['No Deposit','Non Refund','Refundable'])

lead_time = st.slider("How many days elapsed between the  booking date and the arrival date",0,800)

adr=st.slider("What is the Average Daily rate of customer ?",0,6000)

arrival_date_day_of_month= st.slider("On which day of the month is customer expected to arrive",1,31)

arrival_date_week_number=st.slider("Kindly input the week of the year",1,52)

total_of_special_requests=st.slider("How many guest in Total are we expecting",1,10)

agent=st.slider("what's the booking agent id",1,535)

market_segment=st.selectbox("Kindly choose the market segement client belongs to",['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate','Complementary', 'Aviation', 'Undefined'])

previous_cancellations=st.slider("How many times has this customer cancelled a booking",0,26)

stays_in_week_nights=st.slider("How many nights is customer staying",0,50)


def prediction():
        row=np.array([deposit_type, lead_time, adr, arrival_date_day_of_month,arrival_date_week_number, total_of_special_requests,agent,market_segment,previous_cancellations,stays_in_week_nights])
        X=DataFrame([row],columns=cols)
        #st.table(X)
        pred=model.predict(X)
        if pred[0]==0:
            st.success('Customer will not Cancell Booking :thumbsup:')
        else:
            st.error('Customer will cancel Booking :cry:')
trigger=st.button('Predict',on_click=prediction)






