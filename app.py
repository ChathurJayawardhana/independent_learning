import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import urllib.request

st.set_page_config(page_title="SL House Price Predictor", page_icon="🏠", layout="wide")

# ── Download model if not present (use this if the .pkl is hosted externally
#    instead of committed to GitHub, e.g. because it's too large) ──
MODEL_PATH = "house_price_model.pkl"
MODEL_URL  = "https://huggingface.co/chathura9798/sl-house-price-model/resolve/main/house_price_model.pkl?download=true"  # e.g. Hugging Face / direct Drive link

if not os.path.exists(MODEL_PATH) and MODEL_URL != "PASTE_YOUR_DIRECT_DOWNLOAD_LINK_HERE":
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ───────────────────────── GLOBAL STYLES ─────────────────────────
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0A2342, #1B998B);
        padding: 2.2rem 1rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(10,35,66,0.25);
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
    .card {
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        background-color: #fafafa;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }
    .card h4 {
        margin-top: 0;
        color: #0A2342;
    }
    .type-badge {
        background-color: #eef7f6;
        border-left: 4px solid #1B998B;
        padding: 0.7rem 0.9rem;
        border-radius: 8px;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        color: #0A2342;
    }
    .type-badge b { color: #1B998B; }
    .metric-card {
        border: 1px solid #e0e0e0;
        border-radius: 14px;
        padding: 1rem;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── HERO HEADER ─────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🏠 Sri Lanka House Price Predictor</h1>
    <p>ML powered price estimation for houses, apartments, annexes & rooms across Sri Lanka</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("house_price_model.pkl")

# Property type is auto-detected from bedrooms / perch / floors,
# using the exact same rule used to label the training data.
PROPERTY_TYPE_INFO = {
    "Room":       "🚪 Single room unit — 1 bedroom on a small land size (perch ≤ 5).",
    "Annex":      "🏚️ Small independent single-floor unit, usually 1-2 bedrooms, on a small-to-medium land size (perch ≤ 15).",
    "Apartment":  "🏢 Multi-floor unit (2+ floors) built on a small-to-medium land size (perch ≤ 20).",
    "Full House": "🏡 Standalone full house — larger land size, or multiple floors on a bigger plot.",
}

def assign_property_type(bedrooms, perch, floors):
    if bedrooms == 1 and perch <= 5:
        return "Room"
    elif bedrooms <= 2 and floors == 1 and perch <= 15:
        return "Annex"
    elif floors >= 2 and perch <= 20:
        return "Apartment"
    else:
        return "Full House"

tab1, tab2 = st.tabs(["🔮 Predict Price", "📊 Analysis"])

# ── TAB 1: PREDICT ──
with tab1:
    st.markdown('<div class="card"><h4>Enter House Details</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox("District", ['Colombo','Gampaha','Kandy','Galle','Matara',
            'Kalutara','Kurunegala','Ratnapura','Anuradhapura','Polonnaruwa',
            'Badulla','Batticaloa','Trincomalee','Jaffna','Vavuniya',
            'Hambantota','Nuwara Eliya','Matale','Ampara','Puttalam',
            'Kegalle','Moneragala','Mannar','Kilinochchi','Mullaitivu'])
        perch         = st.slider("Land Size (Perch)", 1, 80, 10)
        bedrooms      = st.slider("Bedrooms", 1, 7, 3)
        bathrooms     = st.slider("Bathrooms", 1, 6, 2)
        floors        = st.slider("Floors", 1, 5, 1)

    with col2:
        kitchen_area   = st.slider("Kitchen Area (sqft)", 30, 500, 150)
        parking_spots  = st.slider("Parking Spots", 0, 5, 1)
        year_built     = st.slider("Year Built", 1985, 2025, 2015)
        has_garden     = st.checkbox("Has Garden")
        has_ac         = st.checkbox("Has AC")
        water_supply   = st.selectbox("Water Supply", ['Pipe-borne', 'Well', 'Both'])
        electricity    = st.selectbox("Electricity", ['Single phase', 'Three phase'])

    # auto-detected property type, based on the current slider values above
    property_type = assign_property_type(bedrooms, perch, floors)
    st.markdown(
        f'<div class="type-badge">🔎 Detected Property Type: <b>{property_type}</b><br>{PROPERTY_TYPE_INFO[property_type]}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        house_age      = 2025 - year_built
        bed_bath_ratio = bedrooms / (bathrooms + 1)
        total_rooms    = bedrooms + bathrooms
        is_modern      = 1 if year_built >= 2010 else 0
        large_land     = 1 if perch >= 20 else 0
        luxury         = 1 if (has_ac and has_garden) else 0
        perch_x_floors = perch * floors

        input_df = pd.DataFrame([{
            'perch': perch, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'kitchen_area_sqft': kitchen_area, 'parking_spots': parking_spots,
            'has_garden': int(has_garden), 'has_ac': int(has_ac), 'floors': floors,
            'house_age': house_age, 'bed_bath_ratio': bed_bath_ratio,
            'total_rooms': total_rooms, 'is_modern': is_modern,
            'large_land': large_land, 'luxury': luxury, 'perch_x_floors': perch_x_floors,
            'district': district, 'water_supply': water_supply,
            'electricity': electricity, 'property_type': property_type
        }])

        pred = np.expm1(model.predict(input_df)[0])

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1B998B,#0A2342);
                    padding:2rem;border-radius:16px;text-align:center;color:white;margin-top:1rem;
                    box-shadow:0 8px 24px rgba(10,35,66,0.25);'>
            <p style='font-size:1rem;opacity:0.8'>Estimated Price</p>
            <p style='font-size:3rem;font-weight:800'>LKR {pred:,.0f}</p>
            <p style='font-size:1.2rem'>≈ {pred/1e6:.2f} Million LKR</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: ANALYSIS ──
with tab2:
    df = pd.read_csv("house_prices_srilanka.csv")
    Q1 = df['price_lkr'].quantile(0.25)
    Q3 = df['price_lkr'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price_lkr'] >= Q1-3*IQR) & (df['price_lkr'] <= Q3+3*IQR)].copy()

    # derive property_type for the whole dataset using the same auto-detect rule
    df['property_type'] = df.apply(
        lambda r: assign_property_type(r['bedrooms'], r['perch'], r['floors']), axis=1
    )
    df['price_per_perch'] = df['price_lkr'] / df['perch']

    # ── KPI Row ──
    st.markdown('<div class="card"><h4>Overview</h4>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in zip(
        [k1, k2, k3, k4],
        ["Total Listings", "Median Price", "Avg Price / Perch", "Districts Covered"],
        [f"{len(df):,}", f"{df['price_lkr'].median()/1e6:.1f}M LKR",
         f"{df['price_per_perch'].mean()/1e3:,.0f}K LKR", f"{df['district'].nunique()}"]
    ):
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── District price + Property type distribution ──
    st.markdown('<div class="card"><h4>District & Property Type Breakdown</h4>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_district = px.bar(
            df.groupby('district')['price_lkr'].median().sort_values(ascending=False).reset_index(),
            x='district', y='price_lkr',
            title="Median Price by District",
            labels={'price_lkr': 'Price (LKR)', 'district': 'District'}
        )
        fig_district.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_district, use_container_width=True)
    with c2:
        fig_type = px.pie(
            df, names='property_type', title="Property Type Share", hole=0.45
        )
        fig_type.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_type, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Price distribution + Perch vs Price ──
    st.markdown('<div class="card"><h4>Price Patterns</h4>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = px.histogram(
            df, x='price_lkr', nbins=40, title="Price Distribution",
            labels={'price_lkr': 'Price (LKR)'}
        )
        fig_hist.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
    with c4:
        fig_scatter = px.scatter(
            df, x='perch', y='price_lkr', color='property_type',
            title="Land Size vs Price", opacity=0.6,
            labels={'perch': 'Perch', 'price_lkr': 'Price (LKR)'}
        )
        fig_scatter.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Price per perch by property type + amenities impact ──
    st.markdown('<div class="card"><h4>What Drives Price?</h4>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig_ppp = px.box(
            df, x='property_type', y='price_per_perch', color='property_type',
            title="Price per Perch by Property Type",
            labels={'price_per_perch': 'Price per Perch (LKR)', 'property_type': 'Property Type'}
        )
        fig_ppp.update_layout(showlegend=False, margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_ppp, use_container_width=True)
    with c6:
        amenity_df = pd.DataFrame({
            'Amenity': ['Has Garden', 'No Garden', 'Has AC', 'No AC'],
            'Median Price (M LKR)': [
                df[df['has_garden'] == True]['price_lkr'].median() / 1e6,
                df[df['has_garden'] == False]['price_lkr'].median() / 1e6,
                df[df['has_ac'] == True]['price_lkr'].median() / 1e6,
                df[df['has_ac'] == False]['price_lkr'].median() / 1e6,
            ]
        })
        fig_amenity = px.bar(
            amenity_df, x='Amenity', y='Median Price (M LKR)',
            title="Amenities Impact on Median Price",
            color='Amenity'
        )
        fig_amenity.update_layout(showlegend=False, margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_amenity, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Explore filtered data ──
    st.markdown('<div class="card"><h4>Explore Listings</h4>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        district_filter = st.multiselect("Filter by District", sorted(df['district'].unique()))
    with f2:
        type_filter = st.multiselect("Filter by Property Type", sorted(df['property_type'].unique()))

    filtered_df = df.copy()
    if district_filter:
        filtered_df = filtered_df[filtered_df['district'].isin(district_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['property_type'].isin(type_filter)]

    st.dataframe(
        filtered_df[['district', 'property_type', 'perch', 'bedrooms', 'bathrooms',
                     'floors', 'has_garden', 'has_ac', 'water_supply', 'electricity', 'price_lkr']]
        .sort_values('price_lkr', ascending=False),
        use_container_width=True, height=300
    )
    st.markdown('</div>', unsafe_allow_html=True)
