import streamlit as st

st.title("Streamlitでユーザー情報入力")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("名前")

with col2:
    age = st.number_input("年齢", min_value=0)

if st.button("送信"):
    st.write(f"{name}さんは {age}歳です。")
