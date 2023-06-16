import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
from code.description_data import train_labels


def get_lat_lon_of_request(search_string):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(search_string) + '?format=json'
    response = requests.get(url).json()
    return response[0]["lat"], response[0]["lon"]


st.title("Map")


@st.cache_data
def load_map_df():
    data = np.zeros((len(train_labels), 2))
    for i, company in enumerate(train_labels):
        print(i, company)
        try:
            lat, lon = get_lat_lon_of_request(company + ", Saarland")
        except Exception as e:
            print(e)
            continue
        data[i, 0] = lat
        data[i, 1] = lon
    coords_df = pd.DataFrame(data, columns=["latitude", "longitude"])
    return pd.merge(pd.DataFrame(train_labels, columns=["labels"]), coords_df, right_index=True, left_index=True)


map_df = load_map_df()
st.map(map_df)
