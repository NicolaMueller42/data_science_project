import streamlit as st
import time
import numpy as np
import pandas as pd
import code.description_data as data
import requests

st.title("Home")
if "submitted" not in st.session_state:
    st.session_state.submitted = False
with st.form("Search"):
    st.text_input("Enter company name:")
    st.text_input("Enter your company description:")
    st.form_submit_button("Submit")

with st.spinner("Analyzing..."):
    time.sleep(3)


def get_lat_lon_of_request(search_string):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(search_string) + '?format=json'
    response = requests.get(url).json()
    return response[0]["lat"], response[0]["lon"]


@st.cache_data
def load_map_df(labels):
    data = np.zeros((len(labels), 2))
    for i, company in enumerate(labels):
        print(i, company)
        try:
            lat, lon = get_lat_lon_of_request(company + ", Saarland")
        except Exception as e:
            print(e)
            continue
        data[i, 0] = lat
        data[i, 1] = lon
    coords_df = pd.DataFrame(data, columns=["latitude", "longitude"])
    return pd.merge(pd.DataFrame(labels, columns=["labels"]), coords_df, right_index=True, left_index=True)


if st.session_state.submitted:
    with st.expander("Competitors", expanded=True):
        competitors = np.random.choice(data.train_labels, 5, replace=False)
        st.write(f"Your competitors are: {competitors}")


