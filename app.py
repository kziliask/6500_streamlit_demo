import streamlit as st
from core import read_data, make_iris_plot, run_model
st.set_page_config(page_title="6500 Streamlit Demo", layout="wide")
st.title("Hello, 6500!")
st.write("Welcome to your first Streamlit app.")
st.write("""
> This app demonstrates loading data, creating interactive widgets, and generating plots.
## And all using Streamlit!
__and also some core functions defined in a separate module.__
         """)
df = read_data()
a = st.slider("Select a number", 0, df.shape[0], 50)
b = st.slider("Select another number", 0.0, float(df.shape[1]), 2., step=0.1)

st.write("The sum of the two numbers is:", a + b)

st.write(df.head(a))
col1, col2 = st.columns(2)
with col1:
    x_col = st.selectbox("Select x column", options=df.columns.tolist())
with col2:
    y_col = st.selectbox("Select y column", options=df.columns.tolist(), index=1)

if st.button("Generate Plot"):
    chart_df = df[df['sepal length (cm)'] < b]
    chart_1 = make_iris_plot(chart_df, x_col=x_col, y_col=y_col)
    st.write(chart_1)
