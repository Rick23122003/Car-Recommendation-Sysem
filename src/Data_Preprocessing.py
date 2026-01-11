import pandas as pd
import os
def load_and_preprocess_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "Dataset", "Cars.csv")
    df=pd.read_csv(csv_path)
    df=df.drop(columns=["S.No.","Location","Kilometers_Driven","Owner_Type"])
    df["Price"]=df["New_Price"].combine_first(df["Price"])
    df=df.drop(columns=["New_Price"])
    df["Mileage"]=df["Mileage"].astype(str).str.extract(r"([\d\.]+)").astype(float)
    df["Mileage"]=df["Mileage"].fillna(df["Mileage"].median())
    # print(df["Mileage"].isna().sum())
    df["Engine"]=df["Engine"].astype(str).str.extract(r"([\d\.]+)").astype(float)
    df["Engine"]=df["Engine"].fillna(df["Engine"].median())

    df["Seats"]=df["Seats"].fillna(df["Seats"].median())

    df["Power"]=df["Power"].astype(str).str.extract(r"([\d\.]+)").astype(float)
    df["Power"]=df["Power"].fillna(df["Power"].median())

    df=df.dropna(subset=["Price"])
    # print(df.isna().sum())
    df["Price"]=df["Price"].astype(str).str.extract(r"([\d\.]+)").astype(float)

    return df