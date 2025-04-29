import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
df = pd.read_csv("smartphones_cleaned_v6.csv")

# Fill missing values
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

# Common specs for recommendations - these will be the most relevant features for users
key_features = ['price', 'ram_capacity', 'internal_memory', 'battery_capacity', 'screen_size']


def recommend_phones(specifications, top_n=5):
  
    # Calculate weighted distance for each phone
    distances = []

    for idx, row in df.iterrows():
        distance = 0
        for feature, target_value in specifications.items():
            # Normalize the difference by feature range to make features comparable
            feature_range = df[feature].max() - df[feature].min()
            if feature_range == 0:  # Avoid division by zero
                feature_range = 1

            # Calculate normalized distance for this feature
            feature_distance = abs(row[feature] - target_value) / feature_range

            distance += feature_distance

        distances.append((idx, distance))

    # Sort by smallest distance (most similar)
    distances.sort(key=lambda x: x[1])

    # Get the top_n phones
    top_indices = [idx for idx, _ in distances[:top_n]]
    recommended_phones = df.loc[top_indices].copy()

    return recommended_phones


# Streamlit UI
st.set_page_config(page_title="Smartphone Finder", layout="wide")

st.title("📱 Smartphone Recommendation System")
st.markdown("### Find your perfect smartphone based on specifications")

# Create columns for the UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Set Your Preferences")

    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    target_price = st.slider("Price (PKR)", min_price, max_price,
                             value=int((min_price + max_price) / 2),
                             step=1000)

    ram_options = sorted(df['ram_capacity'].unique())
    target_ram = st.selectbox("RAM (GB)", ram_options,
                              index=len(ram_options) // 2)

    memory_options = sorted(df['internal_memory'].unique())
    target_memory = st.selectbox("Storage (GB)", memory_options,
                                 index=len(memory_options) // 2)

    min_battery = int(df['battery_capacity'].min())
    max_battery = int(df['battery_capacity'].max())
    target_battery = st.slider("Battery Capacity (mAh)", min_battery, max_battery,
                               value=int((min_battery + max_battery) / 2),
                               step=100)

    min_screen = float(df['screen_size'].min())
    max_screen = float(df['screen_size'].max())
    target_screen = st.slider("Screen Size (inches)", min_screen, max_screen,
                              value=float((min_screen + max_screen) / 2),
                              step=0.1)

    top_n = st.slider("Number of recommendations", 1, 10, 5)

    # Create specifications dictionary
    specifications = {
        'price': target_price,
        'ram_capacity': target_ram,
        'internal_memory': target_memory,
        'battery_capacity': target_battery,
        'screen_size': target_screen
    }

    find_button = st.button("Find Phones", type="primary")

# Show recommendations
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if find_button:
    with st.spinner("Finding the best matches..."):
        recommendations = recommend_phones(specifications, top_n)
        st.session_state.recommendations = recommendations

with col2:
    if st.session_state.recommendations is not None:
        st.subheader("Recommended Smartphones")

        for i, (idx, row) in enumerate(st.session_state.recommendations.iterrows()):
            with st.expander(f"#{i + 1}: {row['brand_name']} {row['model']} - PKR {int(row['price']):,}"):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**Brand:** {row['brand_name']}")
                    st.markdown(f"**Model:** {row['model']}")
                    st.markdown(f"**Price:** PKR {int(row['price']):,}")
                    st.markdown(f"**Rating:** {row['rating']:.1f}/5")
                    st.markdown(f"**OS:** {row['os']}")

                with col_b:
                    st.markdown(f"**RAM:** {row['ram_capacity']} GB")
                    st.markdown(f"**Storage:** {row['internal_memory']} GB")
                    st.markdown(f"**Battery:** {int(row['battery_capacity'])} mAh")
                    st.markdown(f"**Screen Size:** {row['screen_size']:.1f} inches")
                    st.markdown(f"**Processor:** {row['processor_brand']}")

                # Show match details
                st.markdown("#### Match Details")
                match_data = {}
                for feature in key_features:
                    feature_value = row[feature]
                    target_value = specifications[feature]
                    if feature == 'price':
                        feature_value = f"PKR {int(feature_value):,}"
                        target_value = f"PKR {int(target_value):,}"
                    elif feature == 'screen_size':
                        feature_value = f"{feature_value:.1f} inches"
                        target_value = f"{target_value:.1f} inches"
                    elif feature == 'battery_capacity':
                        feature_value = f"{int(feature_value)} mAh"
                        target_value = f"{int(target_value)} mAh"

                    match_data[feature] = [target_value, feature_value]

                match_df = pd.DataFrame(match_data, index=['Your Preference', 'This Phone'])
                st.dataframe(match_df.T, use_container_width=True)

        # Show comparison of all recommendations
        st.subheader("Compare All Recommendations")
        comparison_df = st.session_state.recommendations[['brand_name', 'model', 'price', 'ram_capacity',
                                                          'internal_memory', 'battery_capacity', 'screen_size']]
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Set your preferences and click 'Find Phones' to get recommendations.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This smartphone recommendation system helps you find the 
    perfect phone based on your preferred specifications.

    Simply set your preferences for key features like price,
    RAM, storage, battery capacity, and screen size to get
    personalized recommendations.
    """
)

# Add dataset statistics
st.sidebar.title("Dataset Stats")
st.sidebar.markdown(f"**Total phones:** {len(df)}")
st.sidebar.markdown(f"**Brands:** {len(df['brand_name'].unique())}")
st.sidebar.markdown(f"**Price range:** PKR {int(df['price'].min()):,} - PKR {int(df['price'].max()):,}")
