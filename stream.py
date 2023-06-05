from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import TimeSeriesModelCreator
import joblib

st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title(':red[Network bandwidth predictor]')
model_obj = TimeSeriesModelCreator()

st.header(":green[Sample dataframes for input csv file]")
clas = st.radio(
"Choose class",
('11', '12', '13', '14', '15', '16', '17', '18', '19', '20'), horizontal=True)

# download sample dataset
df = pd.read_csv(f"datasets/samples/sample{clas}.csv", delimiter=',')

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)
st.dataframe(df)

st.download_button(
   "Press to Download",
   csv,
   f"sample{clas}.csv",
   "text/csv",
   key='download-csv'
)

st.header(":blue[Using model]")
csv_file_buffer = st.file_uploader("Upload a csv file", type=["csv"])
if csv_file_buffer is not None:
    data = pd.read_csv(csv_file_buffer, delimiter=',')
    source = list(data.source.unique())[0]
    destination = list(data.destination.unique())[0]
    input = data[(data.source == source) & (data.destination == destination)][['bandwidth']]
    target = data[(data.source == source) & (data.destination == destination)][['bandwidth']]

st.write("Choose models you want to predict:")
option_50 = st.checkbox('LSTM with 50 nodes models')
option_200 = st.checkbox('LSTM with 200 nodes models')
option_450 = st.checkbox('LSTM with 450 nodes models')
try:
    if option_50:
        scaler50 = joblib.load(f'scalers/scaler50_{source}.save')
        lstm50 = load_model(f'models/lstm50_{source}.h5')
    if option_200:
        scaler200 = joblib.load(f'scalers/scaler200_{source}.save')
        lstm200 = load_model(f'models/lstm200_{source}.h5')
    if option_450:
        scaler450 = joblib.load(f'scalers/scaler450_{source}.save')
        lstm450 = load_model(f'models/lstm450_{source}.h5')
except:
    st.write("You must upload csv file!")

if st.button('Predict'):
    if csv_file_buffer is not None:
        if option_50:
            prediction50 = model_obj.predict(lstm50, scaler50, input)
            fig, ax = plt.subplots()
            ax.plot(target.values)
            ax.plot(pd.DataFrame(prediction50), color = 'red')
            st.pyplot(fig)
        if option_200:
            prediction200 = model_obj.predict(lstm200, scaler200, input)
            fig, ax = plt.subplots()
            ax.plot(target.values)
            ax.plot(pd.DataFrame(prediction200), color = 'red')
            st.pyplot(fig)
        if option_450:
            prediction450 = model_obj.predict(lstm450, scaler450, input)
            fig, ax = plt.subplots()
            ax.plot(target.values)
            ax.plot(pd.DataFrame(prediction450), color = 'red')
            st.pyplot(fig)
    else:
        st.write("You must upload csv file!")

