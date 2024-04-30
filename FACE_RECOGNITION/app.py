import streamlit as st
import pandas as pd
import datetime
import time

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
df = pd.read_csv("Attendence/Attendence_"+ date +".csv")

st.dataframe(df.style.highlight_max(axis=0))