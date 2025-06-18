# Enhanced House Price Predictor with Feature Matching Fix
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# ======================
# APP CONFIGURATION (MUST BE FIRST)
# ======================
st.set_page_config(
    page_title="Premium Home Value Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# DATA & MODEL FUNCTIONS
# ======================
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df  # Remove synthetic features to match original feature space

@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    evaluations = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        evaluations[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    return models, evaluations, X_train.columns.tolist()

# Initialize models, evaluations and original features
models, evaluations, original_features = train_models()

# ======================
# THEME CONFIGURATION
# ======================
def set_theme(theme_name):
    if theme_name == "Dark":
        primaryColor = "#4a6fa5"
        backgroundColor = "#0e1117"
        secondaryBackgroundColor = "#1a1d24"
        textColor = "#f0f2f6"
        font = "sans serif"
        
        card_style = f"""
            background-color: {secondaryBackgroundColor};
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border-left: 4px solid {primaryColor};
            margin-bottom: 20px;
        """
    else:  # Light theme
        primaryColor = "#4a6fa5"
        backgroundColor = "#f8f9fa"
        secondaryBackgroundColor = "#ffffff"
        textColor = "#333333"
        font = "sans serif"
        
        card_style = f"""
            background-color: {secondaryBackgroundColor};
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-left: 4px solid {primaryColor};
            margin-bottom: 20px;
        """
    
    theme_css = f"""
    <style>
        :root {{
            --primary: {primaryColor};
            --secondary: #166088;
            --accent: #4fc1e9;
            --background: {backgroundColor};
            --card: {secondaryBackgroundColor};
            --text: {textColor};
        }}
        
        .stApp {{
            background-color: var(--background);
            color: var(--text);
            font-family: {font};
        }}
        
        .stButton>button {{
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 12px 28px;
            border: none;
            transition: all 0.3s;
            font-weight: 500;
        }}
        
        .stButton>button:hover {{
            background-color: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .metric-card {{
            {card_style}
        }}
        
        .feature-card {{
            background-color: var(--card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .stSlider>div>div>div>div {{
            background-color: var(--primary) !important;
        }}
        
        .stSelectbox>div>div>div {{
            border-color: var(--primary) !important;
        }}
        
        h1, h2, h3 {{
            color: var(--primary) !important;
        }}
        
        .stProgress>div>div>div>div {{
            background-color: var(--accent) !important;
        }}
        
        .highlight-card {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 12px;
            padding: 25px;
            color: white !important;
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
            border: none;
            margin-bottom: 20px;
        }}
        
        .highlight-card h1, .highlight-card h3 {{
            color: white !important;
        }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

# ======================
# MAIN APP FUNCTION
# ======================
def main():
    # Initialize session state for theme
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    
    # Load data
    df = load_data()
    dark_mode = st.session_state.theme == "Dark"
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=HomeValue+Pro", width=150)
        
        # Theme toggle
        st.markdown("### Theme Settings")
        current_theme = st.radio(
            "Select Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key="theme_selector",
            horizontal=True
        )
        
        if current_theme != st.session_state.theme:
            st.session_state.theme = current_theme
            st.rerun()
        
        set_theme(st.session_state.theme)
        
        st.title("Configuration")
        selected_model = st.selectbox(
            "Choose Prediction Model",
            list(models.keys()),
            help="Select which machine learning model to use for predictions"
        )
        
        st.markdown("---")
        st.markdown("### Model Performance")
        for name, metrics in evaluations.items():
            with st.expander(f"{name} Metrics"):
                st.metric("Mean Absolute Error", f"{metrics['MAE']:.3f}")
                st.metric("R² Score", f"{metrics['R2']:.3f}")
        
        st.markdown("---")
        st.markdown("**About**")
        st.info("""
        This premium tool estimates California home values using advanced machine learning.
        Values are in $100,000s (median for neighborhood blocks).
        """)
    
    # Main content area
    st.title("🏡 Premium Home Value Estimator")
    st.markdown("""
    <div style='background-color: var(--card); padding:20px; border-radius:10px; margin-bottom:20px;'>
    <h3 style='color: var(--primary);'>How to Use</h3>
    <ol>
        <li>Adjust the feature values using the interactive controls</li>
        <li>Select your preferred prediction model from the sidebar</li>
        <li>Click "Estimate Value" to see the prediction and analysis</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Neighborhood Characteristics")
        inputs = {}
        
        with st.container():
            st.markdown("#### Demographic Factors")
            inputs['MedInc'] = st.slider(
                "Median Income (tens of thousands)",
                float(df['MedInc'].min()),
                float(df['MedInc'].max()),
                float(df['MedInc'].median()),
                step=0.1
            )
            inputs['Population'] = st.slider(
                "Population (hundreds)",
                float(df['Population'].min()),
                float(df['Population'].max()),
                float(df['Population'].median()),
                step=1.0
            )
            inputs['AveOccup'] = st.slider(
                "Average Occupancy",
                float(df['AveOccup'].min()),
                float(df['AveOccup'].max()),
                float(df['AveOccup'].median()),
                step=0.1
            )
        
        with st.container():
            st.markdown("#### Property Characteristics")
            inputs['HouseAge'] = st.slider(
                "Median House Age (years)",
                float(df['HouseAge'].min()),
                float(df['HouseAge'].max()),
                float(df['HouseAge'].median()),
                step=1.0
            )
            inputs['AveRooms'] = st.slider(
                "Average Rooms",
                float(df['AveRooms'].min()),
                float(df['AveRooms'].max()),
                float(df['AveRooms'].median()),
                step=0.1
            )
            inputs['AveBedrms'] = st.slider(
                "Average Bedrooms",
                float(df['AveBedrms'].min()),
                float(df['AveBedrms'].max()),
                float(df['AveBedrms'].median()),
                step=0.1
            )
    
    with col2:
        st.subheader("Geographic Location")
        with st.container():
            st.markdown("#### Coordinates")
            col_lat, col_lon = st.columns([1, 1])
            with col_lat:
                inputs['Latitude'] = st.slider(
                    "Latitude",
                    float(df['Latitude'].min()),
                    float(df['Latitude'].max()),
                    float(df['Latitude'].median()),
                    step=0.01,
                    format="%.2f"
                )
            with col_lon:
                inputs['Longitude'] = st.slider(
                    "Longitude",
                    float(df['Longitude'].min()),
                    float(df['Longitude'].max()),
                    float(df['Longitude'].median()),
                    step=0.01,
                    format="%.2f"
                )
            
            st.markdown("#### Location Preview")
            map_data = pd.DataFrame({
                'lat': [inputs['Latitude']],
                'lon': [inputs['Longitude']]
            })
            st.map(map_data, zoom=6)
        
        if st.button("🚀 Estimate Value", key="predict_button", use_container_width=True):
            with st.spinner("Analyzing property characteristics..."):
                time.sleep(1)
                
                # Create input DataFrame with only the original features
                input_df = pd.DataFrame([inputs])[original_features]
                model = models[selected_model]
                prediction = model.predict(input_df)[0]
                scaled_prediction = prediction * 100000
                
                # Calculate derived metrics for display only
                price_per_room = scaled_prediction / inputs['AveRooms']
                price_per_bedroom = scaled_prediction / inputs['AveBedrms']
                
                st.success("Analysis Complete!")
                
                # Enhanced result cards
                st.markdown(f"""
                <div class='highlight-card'>
                    <h3>Estimated Property Value</h3>
                    <h1>${scaled_prediction:,.2f}</h1>
                    <p>Median home value for similar properties in this area</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_metric1, col_metric2 = st.columns([1, 1])
                with col_metric1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Price Per Room</h3>
                        <h2>${price_per_room:,.2f}</h2>
                        <p>Based on {inputs['AveRooms']} rooms</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metric2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Price Per Bedroom</h3>
                        <h2>${price_per_bedroom:,.2f}</h2>
                        <p>Based on {inputs['AveBedrms']} bedrooms</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature analysis
                st.subheader("Feature Analysis")
                fig = px.bar(
                    pd.DataFrame({
                        'Feature': original_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='blues' if not dark_mode else 'deep'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    font=dict(color='white' if dark_mode else 'black')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.download_button(
                    label="📥 Download Prediction Report",
                    data=input_df.to_csv(index=False),
                    file_name="home_value_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()