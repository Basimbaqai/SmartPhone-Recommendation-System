import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("smartphones_cleaned_v6.csv")

df['rating'].fillna(df['rating'].median(), inplace=True)
df['processor_brand'].fillna(df['processor_brand'].mode()[0], inplace=True)
df['num_cores'].fillna(df['num_cores'].median(), inplace=True)
df['fast_charging'].fillna(df['fast_charging'].median(), inplace=True)
df['primary_camera_front'].fillna(df['primary_camera_front'].median(), inplace=True)
df['processor_speed'].fillna(df['processor_speed'].median(), inplace=True)
df['battery_capacity'].fillna(df['battery_capacity'].median(), inplace=True)
df['num_front_cameras'].fillna(df['num_front_cameras'].mode()[0], inplace=True)

df = df.drop(columns=["extended_upto"], errors="ignore")

original_df = df.copy()

numerical_features = ['price', 'rating', 'processor_speed', 'battery_capacity',
                      'ram_capacity', 'internal_memory', 'screen_size', 'refresh_rate',
                      'num_rear_cameras', 'num_front_cameras', 'primary_camera_rear',
                      'primary_camera_front', 'resolution_width']

categorical_features = ['brand_name', 'processor_brand', 'os']

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                      columns=encoder.get_feature_names_out(categorical_features),
                                      index=df.index)

sc = StandardScaler()
scaled_numerical = sc.fit_transform(df[numerical_features])
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features)

df_processed = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

df_final = df_processed.values


def recommend_by_feature(feature, phone_index, top_n=5, original_df=None):
    if original_df is None:
        raise ValueError("Original DataFrame must be provided")

    if feature not in original_df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset!")

    query_value = original_df.loc[phone_index, feature]

    similarity_df = original_df.copy()
    similarity_df["similarity"] = np.abs(similarity_df[feature] - query_value)

    recommended_df = similarity_df.nsmallest(top_n + 1, 'similarity').iloc[1:].copy()

    return recommended_df


st.title("📱 Smartphone Recommendation System")
st.sidebar.header("User Selection")

phone_options = [f"{idx} - {row['model']} (PKR{row['price']})" for idx, row in df.iterrows()]
selected_phone_option = st.sidebar.selectbox("Select a phone:", phone_options)

phone_index = int(selected_phone_option.split(" - ")[0])

st.write(f"### Selected Phone Details")
st.write(df.loc[[phone_index]].T)

feature = st.sidebar.selectbox("Select feature to recommend based on:", numerical_features)

top_n = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

if st.sidebar.button("Recommend"):
    recommended_phones = recommend_by_feature(
        feature=feature,
        phone_index=phone_index,
        top_n=top_n,
        original_df=df
    )

    st.write(f"## Recommended Phone Details")
    st.dataframe(recommended_phones.T)
