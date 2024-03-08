import streamlit as st
import pandas as pd
from model.recommender import Recommender

st.set_page_config(layout="wide")

recommender = Recommender()

df_interaction = pd.read_csv('dataset/new/purchase_history.csv')
df_items = pd.read_csv('dataset/new/product_details.csv')
df_user = pd.read_csv('dataset/new/customer_interactions.csv')

# print(df_interaction)
# print(df_items)
# print(df_user)

recommender.preprocess(df_user, df_items, df_interaction)
recommender.fit_dataset()
recommender.build_dataset()
recommender.load_model('model/lightfm.pickle')

user_id_list = df_user.customer_id.unique().tolist()

st.title("Customer Next Purchase Prediction")
user_id = st.selectbox("Select Customer Id", user_id_list)


history, recommend = recommender.predict(user_id)

history = history.merge(df_interaction, how='inner', on=['product_id'])
history = history[['product_id', 'category', 'price', 'ratings', 'purchase_date']]

# print(history)
# print(df_interaction)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### User Last Purchases")
    st.dataframe(history)
with col2:
    st.markdown("### User 10 Most Likely Like (Next Purchase)")
    st.dataframe(recommend)
