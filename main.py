import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv("/content/insurance.csv")
df.replace({'sex':{'male':0,'female':1}},inplace=True)
df.replace({'smoker':{'yes':1,'no':0}},inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)
x=df.drop(columns='charges',axis=1)
y=df['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=LinearRegression()
model.fit(x_train,y_train)
st.title("Insurance Cost Prediction App")
st.write("This Streamlit app predicts insurance charges based on user inputs.")
# Sidebar input fields
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=44)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=67.0)
children = st.sidebar.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# Encoding user input
sex_val = 0 if sex == 'male' else 1
smoker_val = 1 if smoker == 'yes' else 0
region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
region_val = region_map[region]

# Prediction
input_data = np.array([age, sex_val, bmi, children, smoker_val, region_val]).reshape(1, -1)

if st.sidebar.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.subheader(f"Estimated Insurance Cost: ${prediction[0]:.2f}")

# Show model performance
st.markdown("### Model Performance")
st.write(f"Training R² Score: **{train_r2:.2f}**")
st.write(f"Testing R² Score: **{test_r2:.2f}**")
