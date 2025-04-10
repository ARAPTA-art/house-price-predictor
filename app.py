from pathlib import Path
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from typing import Tuple, Optional
import logging
from datetime import datetime

# Streamlit configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative paths for deployment compatibility
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgboost_pipeline.pkl"
TRAIN_DATA_PATH = BASE_DIR / "train.csv"
CURRENT_YEAR = datetime.now().year
USD_TO_KSH = 130  # Fixed exchange rate: 1 USD = 130 KSh

# Custom CSS for UI
st.markdown("""
    <style>
    :root {
        --primary-color: #4a6baf; --secondary-color: #f8f9fa; --text-color: #343a40; --shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    @media (prefers-color-scheme: dark) {
        --primary-color: #7b9bff; --secondary-color: #2d2d2d; --text-color: #ffffff; --shadow: 0 4px 6px rgba(255,255,255,0.1);
    }
    .header { font-size: 2rem; font-weight: 700; color: var(--text-color); margin-bottom: 1.5rem; }
    .subheader { font-size: 1.5rem; font-weight: 600; color: var(--text-color); margin-bottom: 1rem; }
    .metric-card { background-color: var(--secondary-color); border-radius: 0.625rem; padding: 1rem; margin-bottom: 1rem; box-shadow: var(--shadow); }
    .metric-title { font-size: 0.875rem; color: #6c757d; margin-bottom: 0.3125rem; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: var(--text-color); }
    .stButton>button { background-color: var(--primary-color); color: white; border-radius: 0.625rem; padding: 0.5rem 1rem; border: none; }
    .stButton>button:hover { background-color: #3a5a9f; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(ttl=3600)
def load_model(path: Path) -> Optional[Pipeline]:
    try:
        if not path.exists():
            st.warning(f"Model file {path} not found. Creating a new pipeline.")
            return None
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data(ttl=3600)
def load_training_data(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            st.warning(f"Training data {path} not found.")
            return None
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        st.error(f"Failed to load training data: {e}")
        return None

def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'Neighborhood',
                'OverallQual', 'YearBuilt', 'LotArea', 'GarageCars', 'Fireplaces', 'KitchenQual']
    if is_train and 'SalePrice' in df.columns:
        features.append('SalePrice')
    df = df[features].copy()
    fill_values = {
        'GrLivArea': df['GrLivArea'].median() if not df['GrLivArea'].isna().all() else 0,
        'BedroomAbvGr': df['BedroomAbvGr'].median() if not df['BedroomAbvGr'].isna().all() else 0,
        'FullBath': df['FullBath'].median() if not df['FullBath'].isna().all() else 0,
        'HalfBath': 0, 'OverallQual': df['OverallQual'].median() if not df['OverallQual'].isna().all() else 5,
        'YearBuilt': df['YearBuilt'].median() if not df['YearBuilt'].isna().all() else 2000,
        'LotArea': df['LotArea'].median() if not df['LotArea'].isna().all() else 0,
        'GarageCars': df['GarageCars'].median() if not df['GarageCars'].isna().all() else 0,
        'Fireplaces': 0,
        'KitchenQual': df['KitchenQual'].mode()[0] if not df['KitchenQual'].mode().empty else 'TA'
    }
    df = df.fillna(fill_values)
    df = df.assign(
        TotalBath=lambda x: x['FullBath'] + 0.5 * x['HalfBath'],
        HouseAge=lambda x: CURRENT_YEAR - x['YearBuilt'],
        QualityArea=lambda x: x['GrLivArea'] * x['OverallQual']
    )
    return df

def create_pipeline() -> Pipeline:
    numeric_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'OverallQual', 
                        'YearBuilt', 'LotArea', 'GarageCars', 'Fireplaces', 'TotalBath', 
                        'HouseAge', 'QualityArea']
    categorical_features = ['Neighborhood', 'KitchenQual']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
    ]
    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)
    return Pipeline([('preprocessor', preprocessor), ('regressor', stacking_regressor)])

def retrain_model(pipeline: Pipeline, new_data: pd.DataFrame) -> Pipeline:
    try:
        processed_data = preprocess_data(new_data)
        X = processed_data.drop('SalePrice', axis=1)
        y = np.log1p(processed_data['SalePrice'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        st.write(f"Cross-Validation R²: {cv_scores.mean():.2f} (± {cv_scores.std():.2f})")
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Model retraining failed: {e}")
        return pipeline

def validate_inputs(**kwargs) -> None:
    errors = []
    if kwargs['area'] <= 0: errors.append("Living Area must be greater than 0 sq ft.")
    if kwargs['bedrooms'] < 0: errors.append("Bedrooms cannot be negative.")
    if kwargs['overall_qual'] < 1 or kwargs['overall_qual'] > 10: errors.append("Quality must be 1-10.")
    if kwargs['year_built'] > CURRENT_YEAR: errors.append(f"Year Built cannot be after {CURRENT_YEAR}.")
    if errors:
        raise ValueError("\n".join(errors))

def predict_sale_price(**kwargs) -> Tuple[pd.DataFrame, float]:
    pipeline = kwargs.pop('pipeline')
    validate_inputs(**kwargs)
    input_data = pd.DataFrame([kwargs], index=[0]).rename(columns={
        'area': 'GrLivArea', 'bedrooms': 'BedroomAbvGr', 'full_bath': 'FullBath',
        'half_bath': 'HalfBath', 'neighborhood': 'Neighborhood', 'overall_qual': 'OverallQual',
        'year_built': 'YearBuilt', 'lot_area': 'LotArea', 'garage_cars': 'GarageCars',
        'fireplaces': 'Fireplaces', 'kitchen_qual': 'KitchenQual'
    })
    input_data = preprocess_data(input_data, is_train=False)
    pred_log = pipeline.predict(input_data)[0]
    price_usd = np.expm1(pred_log)
    return input_data, price_usd * USD_TO_KSH

def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    y_pred_log = pipeline.predict(X)
    y_pred_usd = np.expm1(y_pred_log)
    y_actual_usd = np.expm1(y)
    y_pred_ksh = y_pred_usd * USD_TO_KSH
    y_actual_ksh = y_actual_usd * USD_TO_KSH
    st.write("Sample Actual vs Predicted (first 5, in KSh):")
    st.write(pd.DataFrame({'Actual': y_actual_ksh[:5], 'Predicted': y_pred_ksh[:5]}))
    return {
        'MAE': mean_absolute_error(y_actual_ksh, y_pred_ksh),
        'RMSE': np.sqrt(mean_squared_error(y_actual_ksh, y_pred_ksh)),
        'R2': r2_score(y_actual_ksh, y_pred_ksh)
    }

def main():
    st.sidebar.title("Settings")
    theme = st.sidebar.selectbox("Theme", ["System", "Light", "Dark"])

    username = os.environ.get('USER', os.environ.get('USERNAME', 'User'))
    st.markdown(f"""
        <div class="header">🏠 House Price Predictor</div>
        <div style="text-align: right; color: var(--text-color);">Welcome, {username}</div>
    """, unsafe_allow_html=True)

    pipeline = load_model(MODEL_PATH)
    if pipeline is None:
        st.info("No pre-trained model found. Initializing a new pipeline.")
        pipeline = create_pipeline()

    train_data = load_training_data(TRAIN_DATA_PATH)
    if train_data is None and not MODEL_PATH.exists():
        st.error("No initial training data or model found. Please upload data to proceed.")
        return

    st.sidebar.subheader("Retrain Model")
    uploaded_file = st.sidebar.file_uploader("Upload new training data (CSV)", type="csv")
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        with st.spinner("Retraining model..."):
            pipeline = retrain_model(pipeline, new_data)
            st.sidebar.success("Model retrained successfully!")
        train_data = new_data

    if train_data is not None:
        train_processed = preprocess_data(train_data)
        X = train_processed.drop('SalePrice', axis=1)
        y = np.log1p(train_processed['SalePrice'])
        train_processed['PredictedSalePrice'] = np.expm1(pipeline.predict(X)) * USD_TO_KSH
        train_processed['SalePrice'] = train_processed['SalePrice'] * USD_TO_KSH
        metrics = evaluate_model(pipeline, X, y)
        cols = st.columns(4)
        metrics_data = [
            ("Total Properties", f"{len(train_processed):,}"),
            ("Average Price", f"KSh {train_processed['SalePrice'].mean():,.0f}"),
            ("Prediction Accuracy", f"{metrics['R2']*100:.1f}%"),
            ("Typical Price Error", f"KSh {metrics['RMSE']:,.0f}")
        ]
        for col, (title, value) in zip(cols, metrics_data):
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔮 Predict Price", "📊 Market Insights"])
    
    with tab1:
        st.markdown('<div class="subheader">Price Prediction</div>', unsafe_allow_html=True)
        # Fixed the ternary operator syntax here
        neighborhoods = train_data['Neighborhood'].unique().tolist() if train_data is not None else ['NAmes']
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                area = st.number_input("Living Area (sq ft)", min_value=0, value=1500, step=10)
                bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
                full_bath = st.number_input("Full Bathrooms", min_value=0, value=2)
                half_bath = st.number_input("Half Bathrooms", min_value=0, value=1)
            with col2:
                neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)
                overall_qual = st.slider("Quality", 1, 10, 7, help="1 = Poor, 10 = Excellent")
                year_built = st.number_input("Year Built", min_value=1800, max_value=CURRENT_YEAR, value=2000)
                lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=9000, step=100)
                garage_cars = st.number_input("Garage Size (cars)", min_value=0, value=2)
                fireplaces = st.number_input("Fireplaces", min_value=0, value=1)
                kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
            submit = st.form_submit_button("Calculate Price")
        
        if submit:
            with st.spinner("Predicting..."):
                try:
                    input_data, price_ksh = predict_sale_price(
                        area=area, bedrooms=bedrooms, full_bath=full_bath, half_bath=half_bath,
                        neighborhood=neighborhood, overall_qual=overall_qual, year_built=year_built,
                        lot_area=lot_area, garage_cars=garage_cars, fireplaces=fireplaces,
                        kitchen_qual=kitchen_qual, pipeline=pipeline
                    )
                    st.success(f"Predicted Price: **KSh {price_ksh:,.0f}**")
                    regressor = pipeline.named_steps['regressor']
                    explainer = shap.TreeExplainer(regressor.named_estimators_['xgb'] if isinstance(regressor, StackingRegressor) else regressor)
                    shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(input_data))
                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    shap_values_ksh = shap_values * USD_TO_KSH
                    fig = px.bar(
                        x=shap_values_ksh[0], y=[name.split('__')[-1] for name in feature_names],
                        orientation='h', title="Feature Impact on Prediction",
                        labels={'x': 'Impact on Price (KSh)', 'y': 'Features'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with tab2:
        st.markdown('<div class="subheader">Market Insights</div>', unsafe_allow_html=True)
        if train_data is not None:
            neighborhood_filter = st.selectbox("Filter by Neighborhood", ["All"] + train_processed['Neighborhood'].unique().tolist())
            filtered_data = train_processed if neighborhood_filter == "All" else train_processed[train_processed['Neighborhood'] == neighborhood_filter]
            fig1 = px.histogram(filtered_data, x='SalePrice', nbins=30, title="Price Distribution", labels={'SalePrice': 'Price (KSh)'})
            st.plotly_chart(fig1, use_container_width=True)
            top_n = filtered_data.groupby('Neighborhood')['SalePrice'].mean().nlargest(10).reset_index()
            fig2 = px.bar(top_n, x='SalePrice', y='Neighborhood', orientation='h', title="Top Neighborhoods by Avg Price", labels={'SalePrice': 'Avg Price (KSh)'})
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = px.scatter(filtered_data, x='SalePrice', y='PredictedSalePrice', title="Predicted vs Actual Prices", 
                              labels={'SalePrice': 'Actual Price (KSh)', 'PredictedSalePrice': 'Predicted Price (KSh)'}, trendline="ols")
            st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()