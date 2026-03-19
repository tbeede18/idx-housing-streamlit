import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
import joblib
from datetime import datetime
from pathlib import Path
import json
from sklearn.neighbors import BallTree 

import os
import streamlit as st
from pathlib import Path

# --- 0. PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="CA House Price Predictor", layout="wide")

# Setting up Pathing
# base_path = Path(__file__).parent.absolute()
PROJECT_ROOT = Path("/Users/23tylerb/Desktop/IDX_Exchange/idx-housing-streamlit")
# print(PROJECT_ROOT)  # Debugging line to check the current directory
DATA_DIR = PROJECT_ROOT / 'models' / 'data'

# 1. Verify files exist before loading
required_files = ['scaler_X.pkl', 'scale_cols.pkl', 'California_School_District_Areas_2024-25.geojson']

for f in required_files:
    file_path = DATA_DIR / f
    if not file_path.exists():
        st.error(f"🚨 MISSING FILE: {f} is NOT at {file_path}")
    else:
        st.write(f"✅ Found: {f}")

# 3. Load them using the Ironclad Path
try:
    scaler = joblib.load(DATA_DIR / 'scaler_X.pkl')
    scale_cols = joblib.load(DATA_DIR / 'scale_cols.pkl')
    gdf_districts = gpd.read_file(DATA_DIR / 'California_School_District_Areas_2024-25.geojson')
    st.success("🎯 All models and GeoData loaded successfully!")
except Exception as e:
    st.error(f"❌ Load Failed: {e}")

# --- 1. INITIALIZE MEMORY (SESSION STATE) ---
# This is how Streamlit remembers the auto-filled data between button clicks
if "auto_lat" not in st.session_state: st.session_state.auto_lat = None
if "auto_lon" not in st.session_state: st.session_state.auto_lon = None
if "auto_zip_price" not in st.session_state: st.session_state.auto_zip_price = None
if "auto_dist_price" not in st.session_state: st.session_state.auto_dist_price = None
if "auto_dist_rest" not in st.session_state: st.session_state.auto_dist_rest = None

# --- 2. LOAD CACHED DATA ---
@st.cache_data
def load_spatial_data():
    gdf_districts = gpd.read_file(DATA_DIR / 'California_School_District_Areas_2024-25.geojson')
    high_schools_gdf = gdf_districts[gdf_districts['DistrictType'].isin(['High', 'Unified'])]
    return high_schools_gdf

@st.cache_resource
def load_model(path):
    if not path.exists(): return None
    return joblib.load(path)

@st.cache_resource
def load_restaurant_tree():
    try:
        # 1. Load the parquet file
        rest_df = pd.read_parquet(DATA_DIR / 'zengtao_restaurants.parquet')
        # 2. Convert coordinates to radians
        coords_rest = np.radians(rest_df[['lat', 'lon']].values)
        # 3. Build and return the tree
        tree = BallTree(coords_rest, metric="haversine")
        return tree
    except Exception as e:
        st.error(f"Could not load restaurant data: {e}")
        return None

restaurant_tree = load_restaurant_tree()

# Load the spatial data for the district lookup using geopandas
districts_map = load_spatial_data()
# Load the Model
MODEL_PATH = PROJECT_ROOT / 'models' / 'house_price_model_v3.pkl'
model = load_model(MODEL_PATH)

# Load JSON dictionaries securely
try:
    with open('models/data/zip_code_map.json', 'r') as f:
        zip_code_map = json.load(f)
    with open('models/data/district_map.json', 'r') as f:
        district_map = json.load(f)
except FileNotFoundError:
    st.error("JSON files missing. Place them in the 'models' folder.")
    zip_code_map = {"Unknown": 0.0}
    district_map = {"Unknown": 0.0}

# --- 3. AUTO-FILL ADDRESS SEARCH ---
st.title("🏡 CA House Price Predictor")

address_input = st.text_input("Enter Property Address", placeholder="e.g. 3551 Trousdale Pkwy, Los Angeles, CA 90089")

if st.button("Search Address"):
    with st.spinner("Locating property and calculating district data..."):
        geolocator = Nominatim(user_agent="usc_housing_ta_app")
        location = geolocator.geocode(address_input, addressdetails=True)
        
        if location:
            lat, lon = location.latitude, location.longitude
            
            # Extract ZIP
            raw_address = location.raw.get('address', {})
            zip_code = raw_address.get('postcode', 'Unknown')
            
            # Spatial Join for District
            point = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs(districts_map.crs)
            joined = gpd.sjoin(point_gdf, districts_map, how="left", predicate="within")
            
            found_district = 'Out of Bounds / Premium'
            if not joined.empty and pd.notna(joined.iloc[0]['DistrictName']):
                found_district = joined.iloc[0]['DistrictName']

            # Lookup encoded values, falling back to your "Unknown" baked-in keys safely
            encoded_zip_price = zip_code_map.get(zip_code, zip_code_map.get('Unknown', 500000.0))
            encoded_district_price = district_map.get(found_district, district_map.get('Unknown', 500000.0))
            
            # --- NEW: CALCULATE RESTAURANT DISTANCE ---
            if restaurant_tree is not None:
                # Convert this specific house's Lat/Lon to radians
                house_rad = np.radians([[lat, lon]])
                # Query the tree for the nearest 1 restaurant
                dist_rad, _ = restaurant_tree.query(house_rad, k=1)
                # Convert back to miles
                dist_miles = dist_rad[0][0] * 3958.8
            else:
                dist_miles = None # Fallback if file is missing
            
            # --- THE MAGIC TRICK ---
            # Update the memory state so the input boxes below auto-populate!
            st.session_state.auto_lat = float(lat)
            st.session_state.auto_lon = float(lon)
            st.session_state.auto_zip_price = float(encoded_zip_price)
            st.session_state.auto_dist_price = float(encoded_district_price)
            st.session_state.auto_dist_rest = float(dist_miles)

            st.success(f"Found! Detected ZIP: {zip_code} | District: {found_district} | Nearest Restaurant: {dist_miles:.2f} mi")
        else:
            st.error("Could not find that address. Please check spelling.")

st.divider()
# --- 4. MANUAL INPUT SECTION ---
st.write("Review auto-filled location data and fill in the physical house details below.")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("📐 Size & Type") # Renamed since Lat/Lon is gone
    living_area = st.number_input("Living Area (sqft)", min_value=200, value=None, placeholder="e.g. 1800")
    lot_size = st.number_input("Lot Size (sqft)", min_value=500, value=None, placeholder="e.g. 7500")
    stories = st.selectbox("Stories", [1, 2, 3], index=None, placeholder="Select stories...")
    
    # Optional: Just show them the encoded prices as read-only text so they know it worked
    st.caption("Background Data:")
    st.write(f"📍 Zip Value: ${st.session_state.auto_zip_price:,.2f}" if st.session_state.auto_zip_price else "📍 Zip Value: Waiting for address...")
    st.write(f"🏫 District Value: ${st.session_state.auto_dist_price:,.2f}" if st.session_state.auto_dist_price else "🏫 District Value: Waiting for address...")
    st.write(f"🍽️ Nearest Restaurant: {st.session_state.auto_dist_rest:.2f} mi" if st.session_state.auto_dist_rest is not None else "🍽️ Nearest Restaurant: Waiting for address...")

with col2:
    st.header("🛏️ Rooms & Build")
    beds = st.number_input("Bedrooms Total", min_value=1, value=None, placeholder="e.g. 3")
    main_level_beds = st.number_input("Main Level Bedrooms", min_value=0, value=None, placeholder="e.g. 1")
    baths = st.number_input("Bathrooms Total", min_value=1.0, value=None, placeholder="e.g. 2")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=None, placeholder="e.g. 2005")
    
    # These are fine to keep defaults, or you can make them None too!
    days_on_market = st.number_input("Days on Market", value=None, placeholder="e.g. 0")

with col3:
    st.header("💰 Features")
    hoa = st.number_input("Monthly HOA ($)", value=0.0)
    parking = st.number_input("Parking Total", min_value=0, value=None, placeholder="e.g. 2")
    garage_spaces = st.number_input("Garage Spaces", min_value=0, value=None, placeholder="e.g. 2")
    
    view = st.checkbox("View?")
    waterfront = st.checkbox("Waterfront?")
    basement = st.checkbox("Basement?")
    pool = st.checkbox("Private Pool?")
    attached_garage = st.checkbox("Attached Garage?")
    fireplace = st.checkbox("Fireplace?")
    new_const = st.checkbox("New Construction?")

st.header("🎨 Finishes")
flooring_types = st.multiselect(
    "Flooring Materials (Select all that apply)",
    ["Carpet", "Vinyl", "Stone", "Bamboo", "Concrete", "Brick", "Laminate", "Tile", "Wood", "Unknown"],
    default=["Unknown"]
)
# --- 5. PREDICTION LOGIC ---
def create_input_df():
# 1. Map Multiple Flooring Selections to OHE columns
    flooring_cols = {
        'HasCarpet': 0, 'HasVinyl': 0, 'HasStone': 0, 'HasBamboo': 0, 
        'HasConcrete': 0, 'HasBrick': 0, 'HasLaminate': 0, 'HasTile': 0, 
        'HasWood': 0, 'HasUnknownFlooring': 0
    }
    
    # Loop through everything they selected in the UI
    for f_type in flooring_types:
        col_name = f"Has{f_type.replace('Unknown', 'UnknownFlooring')}"
        flooring_cols[col_name] = 1

    home_age = datetime.now().year - year_built if year_built else 0
    
    data = {
        'ViewYN': int(view),
        'WaterfrontYN': int(waterfront),
        'BasementYN': int(basement),
        'PoolPrivateYN': int(pool),
        
        # PULL DIRECTLY FROM INVISIBLE MEMORY
        'Latitude': st.session_state.auto_lat,
        'Longitude': st.session_state.auto_lon,
        'Postal_Code_Encoded': st.session_state.auto_zip_price,
        'District_Avg_Price': st.session_state.auto_dist_price,
        
        'LivingArea': living_area,
        'DaysOnMarket': days_on_market,
        'AttachedGarageYN': int(attached_garage),
        'ParkingTotal': parking,
        'YearBuilt': year_built,
        'BathroomsTotalInteger': int(baths) if baths else 0,
        'BedroomsTotal': int(beds) if beds else 0,
        'FireplaceYN': int(fireplace),
        'Stories': stories if stories else 1,
        'MainLevelBedrooms': main_level_beds,
        'NewConstructionYN': int(new_const),
        'GarageSpaces': garage_spaces,
        'LotSizeSquareFeet': lot_size,
        'Home_Age': float(home_age),
        # 'DistWestCoastMi': np.nan,  # ⚠️ TEMPORARY PLACEHOLDER - Replace with actual distance calculation if desired
        **flooring_cols, 
        'Monthly_HOA': hoa,
        'log_HOA': np.log1p(hoa),
        # 'Home_Age^2': home_age**2,
        'Bed_to_Bath': beds / baths if (beds and baths and baths > 0) else 0,
        'Living_Area_to_Bedrooms': living_area / beds if (living_area and beds and beds > 0) else 0,
        'Living_Area_per_Story': living_area / stories if (living_area and stories) else 0,
        'DistNearestRestaurantMi': st.session_state.auto_dist_rest
    }
    return pd.DataFrame([data])

if st.button("Calculate Value", type="primary"):
    # THE SAFETY CATCH: Make sure they searched an address and filled out the blanks!
    if st.session_state.auto_lat is None:
        st.error("🚨 Please search for a valid address first!")
    elif None in [living_area, beds, baths, year_built, lot_size]:
        st.warning("⚠️ Please fill out all the blank housing details before calculating.")
    # elif model:
    #     input_df = create_input_df()
    #     input_df = input_df[model.feature_names_in_]
        
    #     pred = model.predict(input_df)[0]
    #     st.balloons()
    #     st.metric("Estimated Market Value", f"${pred:,.2f}")
    elif model and scaler:
        input_df = create_input_df()
        
        # 1. Ensure columns are in the exact order the model expects
        input_df = input_df[model.feature_names_in_]

        st.write("Unscaled Inputs:")
        st.dataframe(input_df.iloc[:]) 
        
        # 2. SQUISH THE NUMBERS! (Apply the scaler only to the necessary columns)
        input_df[scale_cols] = scaler.transform(input_df[scale_cols].astype('float32'))
        
        # 3. Predict the Log Value
        raw_pred = model.predict(input_df)[0]
        
        # 4. Convert Log back to Real Dollars
        pred_in_dollars = np.expm1(raw_pred) 
        

        # --- 🔍 THE INSPECTOR BLOCK ---
        
        # 1. Show the Raw DataFrame (What the model sees after scaling)
        st.write("Scaled Inputs:")
        st.dataframe(input_df.iloc[:]) 
        
        # ------------------------------
        
        st.balloons()
        st.metric("Estimated Market Value", f"${pred_in_dollars:,.2f}")
    else:
        st.error("Model or Scaler not loaded. Check the 'models/data' folder.")