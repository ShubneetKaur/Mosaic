import streamlit as st
import pandas as pd
from data_generator import make_data
from federated import run_federated, FEATURES, predict

st.set_page_config(page_title='Federated Traffic Prototype',layout='wide')
st.title('Federated Traffic-Flow Prediction')
st.caption('PS-003 Prototype | Local training + FedAvg + communication-cost evaluation')
locations=['Chandigarh-North','Chandigarh-Central','Chandigarh-South','Zirakpur-Corridor']
rounds=st.sidebar.slider('Federated rounds',2,15,6)
if st.button('Run Prototype',type='primary'):
    data={x:make_data(x,seed=100) for x in locations}
    w,b,history=run_federated(data,rounds=rounds)
    st.session_state.update(data=data,w=w,b=b,history=history)
if 'history' not in st.session_state:
    st.info('Click Run Prototype to start local training and federated aggregation.')
    st.code('Traffic Data -> Local Training -> Model Update -> Local/Edge Server -> Central Server -> Global Model -> Local Training -> Repeat -> Traffic Prediction')
    st.stop()
h=pd.DataFrame(st.session_state.history)
st.metric('Final Global MAE',f"{h.iloc[-1].global_mae:.2f}")
st.metric('Total Model-Update Communication',f"{h.communication_bytes.sum()/1024:.2f} KB")
st.line_chart(h.set_index('round')['global_mae'])
st.bar_chart(h.set_index('round')['communication_bytes'])
loc=st.selectbox('Location for prediction',locations)
row=st.session_state.data[loc].iloc[-1]
p=float(predict(row[FEATURES].values.reshape(1,-1),st.session_state.w,st.session_state.b)[0])
st.success(f'Predicted traffic flow at {loc}: {max(0,p):.0f} vehicles per interval')
st.subheader('Prototype Workflow')
st.code('Traffic Data Collection\n        ↓\nLocal Training\n        ↓\nModel Update\n        ↓\nLocal / Edge Server\n        ↓\nCentral Server\n        ↓\nGlobal Model (FedAvg)\n        ↓\nLocal Training ↺ Repeat\n        ↓\nTraffic Prediction\n        ↓\nEmergency Route Support')
