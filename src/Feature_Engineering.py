import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.Data_Preprocessing import load_and_preprocess_data
df=load_and_preprocess_data()
def feature_engineering(df):

    car_names=df["Name"].reset_index(drop=True)
    categorical_data=["Fuel_Type","Transmission"]
    numeric_data=["Mileage","Engine","Power","Price","Seats"]

    encoded_data=pd.get_dummies(df[categorical_data],dtype=int)    #One-Hot encoding
    encoded_data = encoded_data.reset_index(drop=True)

    scaler=MinMaxScaler()
    scaled_data=pd.DataFrame(scaler.fit_transform(df[numeric_data]),columns=numeric_data)   #Numeric values scaled in the range 0-1
    scaled_data=scaled_data.reset_index(drop=True)
    
    X=pd.concat([encoded_data,scaled_data],axis=1)

    return car_names,X
    