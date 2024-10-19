import os
import pickle
import copyreg 
import types
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# Set page configuration
st.set_page_config(page_title="Steam Game Industry Assistant",
                   layout="wide", page_icon="🎮")

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
sales_model = pickle.load(open(f'{working_dir}/Saved_Models/Steam_Developer_Model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Steam Game Industry Analysis',
                          ['Sales Prediction'],
                          menu_icon='controller', 
                          icons=['cash-coin', 'person-workspace'], 
                          default_index=0)
    
import numpy as np
import pandas as pd
df_vgsales = pd.read_csv("C:/Users/Chetan Vemula/Desktop/VS Code Python/Steam_Games_Prediction_And_Recommendation/Datasets/vgsales.csv")
df_vgsales.head()
nan_count_vgsales = df_vgsales.isna().sum()
print(nan_count_vgsales)
df_vgsales.dropna(inplace=True)
df_vgsales
df_developer = pd.read_csv("C:/Users/Chetan Vemula/Desktop/VS Code Python/Steam_Games_Prediction_And_Recommendation/Datasets/developper.csv", encoding = 'latin-1')
df_developer
inactive_developers = df_developer[df_developer['Active'] == 0]['Developer'].tolist()
# 2. Filter df_vgsales to remove inactive developers
df_vgsales_active = df_vgsales[~df_vgsales['Developer'].isin(inactive_developers)]
# (Notice the '~' for negation to remove the inactive ones)

# Print the DataFrame of active developers
df_vgsales_active

# Sales Prediction Page
if selected == 'Sales Prediction':
    st.title('Predict Global Sales for a Publisher')
    publisher_list = df_vgsales_active['Publisher'].unique() 
    publisher = st.selectbox("Select Publisher", publisher_list)
    year = st.number_input("Year to Predict", min_value=2024, max_value=2050, value=2024) 
    if st.button('Predict Sales'):
        if publisher:
            input_data = np.array([year]).reshape(1, -1)
            prediction = sales_model.predict(input_data)
            st.success(f"Predicted sales for {publisher} in {year}: {prediction[0]:.2f} Million Copies")
        else:
            st.warning("Please select a publisher.")