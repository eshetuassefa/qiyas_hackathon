
# =============================================================================
# app.py
# Ethiopian Smallholder Crop-Yield Predictor
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Ethiopian Crop Yield Predictor",
    page_icon="🌾",
    layout="centered",
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = ASSETS_DIR / "final_model.joblib"
PRICE_PATH = ASSETS_DIR / "price_lookup.csv"
WEATHER_PATH = ASSETS_DIR / "weather_lookup.csv"


# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained machine-learning model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_lookup_data():
    """Load price and weather lookup datasets."""
    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Price lookup file was not found:\n{PRICE_PATH}"
        )

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Weather lookup file was not found:\n{WEATHER_PATH}"
        )

    prices_df = pd.read_csv(PRICE_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)

    return prices_df, weather_df


# Load resources
try:
    model = load_model()
    prices, weather = load_lookup_data()

except Exception as error:
    st.error("The application could not load the required files.")
    st.exception(error)
    st.stop()


# =============================================================================
# VALIDATE LOOKUP DATA
# =============================================================================

required_price_columns = {
    "region",
    "crop_type",
    "survey_year",
    "price_birr_per_quintal",
}

required_weather_columns = {
    "region",
    "year",
    "month_num",
    "avg_temp_c",
    "monthly_rainfall_mm",
    "extreme_heat_days",
}


missing_price_columns = required_price_columns - set(prices.columns)
missing_weather_columns = required_weather_columns - set(weather.columns)


if missing_price_columns:
    st.error(
        "The price lookup file is missing the following columns: "
        + ", ".join(sorted(missing_price_columns))
    )
    st.stop()


if missing_weather_columns:
    st.error(
        "The weather lookup file is missing the following columns: "
        + ", ".join(sorted(missing_weather_columns))
    )
    st.stop()


# =============================================================================
# CONSTANTS
# =============================================================================

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


DEFAULT_TEMP = 18.0
DEFAULT_RAINFALL = 600.0
DEFAULT_HEAT_DAYS = 5.0
DEFAULT_PRICE = 4000.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_season_weather(
    weather_df: pd.DataFrame,
    region: str,
    planting_month: int,
    survey_year: int,
):
    """
    Look up four consecutive months of weather beginning with
    the selected planting month.

    Returns:
        season_avg_temp
        season_total_rainfall
        season_extreme_heat_days
    """

    # Four-month growing/production window
    season_months = [
        ((planting_month + i - 1) % 12) + 1
        for i in range(4)
    ]

    # Handle crossing from December into January
    target_years = [
        survey_year if month >= planting_month else survey_year + 1
        for month in season_months
    ]

    region_weather = weather_df[
        weather_df["region"] == region
    ].copy()

    if region_weather.empty:
        return (
            DEFAULT_TEMP,
            DEFAULT_RAINFALL,
            DEFAULT_HEAT_DAYS,
            False,
        )

    # Build the weather mask
    mask = pd.Series(False, index=region_weather.index)

    for month, year in zip(season_months, target_years):
        mask |= (
            (region_weather["year"] == year)
            & (region_weather["month_num"] == month)
        )

    season = region_weather.loc[mask]

    # No matching weather records
    if season.empty:
        return (
            DEFAULT_TEMP,
            DEFAULT_RAINFALL,
            DEFAULT_HEAT_DAYS,
            False,
        )

    season_avg_temp = season["avg_temp_c"].mean()
    season_total_rainfall = season["monthly_rainfall_mm"].sum()
    season_extreme_heat_days = season["extreme_heat_days"].sum()

    return (
        float(season_avg_temp),
        float(season_total_rainfall),
        float(season_extreme_heat_days),
        True,
    )


def get_crop_price(
    prices_df: pd.DataFrame,
    region: str,
    crop_type: str,
    survey_year: int,
):
    """
    Look up the average crop price for the selected
    region, crop and survey year.
    """

    matching_prices = prices_df[
        (prices_df["region"] == region)
        & (prices_df["crop_type"] == crop_type)
        & (prices_df["survey_year"] == survey_year)
    ]

    if matching_prices.empty:
        return DEFAULT_PRICE, False

    price = matching_prices["price_birr_per_quintal"].mean()

    return float(price), True


def get_regional_mean_temperature(
    weather_df: pd.DataFrame,
    region: str,
):
    """Calculate the average temperature for the selected region."""

    regional_weather = weather_df[
        weather_df["region"] == region
    ]

    if regional_weather.empty:
        return DEFAULT_TEMP

    return float(regional_weather["avg_temp_c"].mean())


# =============================================================================
# APPLICATION HEADER
# =============================================================================

st.title("🌾 Ethiopian Smallholder Crop-Yield Predictor")

st.markdown(
    """
This application estimates **crop yield (tons per hectare)** and
**potential farm revenue** using plot characteristics, crop information,
historical weather conditions, and crop prices.
"""
)

st.info(
    "Weather and crop price information are looked up automatically "
    "from the datasets included with the application."
)


# =============================================================================
# INPUT SECTION
# =============================================================================

st.subheader("🌱 Plot and Crop Information")

col1, col2 = st.columns(2)


# -----------------------------------------------------------------------------
# Column 1
# -----------------------------------------------------------------------------

with col1:

    region_options = sorted(
        prices["region"].dropna().unique().tolist()
    )

    crop_options = sorted(
        prices["crop_type"].dropna().unique().tolist()
    )

    region = st.selectbox(
        "Region",
        region_options,
    )

    crop = st.selectbox(
        "Crop type",
        crop_options,
    )

    year = st.selectbox(
        "Survey year",
        [2021, 2022, 2023, 2024],
    )

    planting_month = st.selectbox(
        "Planting month",
        list(MONTH_MAP.keys()),
    )


# -----------------------------------------------------------------------------
# Column 2
# -----------------------------------------------------------------------------

with col2:

    altitude = st.number_input(
        "Altitude (m)",
        min_value=500.0,
        max_value=3500.0,
        value=1800.0,
        step=50.0,
    )

    farm_size = st.number_input(
        "Farm size (ha)",
        min_value=0.1,
        max_value=20.0,
        value=1.0,
        step=0.1,
    )

    fertilizer = st.number_input(
        "Fertilizer (kg/ha)",
        min_value=0.0,
        max_value=300.0,
        value=50.0,
        step=5.0,
    )

    improved_seed = st.selectbox(
        "Improved seed used?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    pest_disease = st.selectbox(
        "Pest/disease flag",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    soil_quality = st.slider(
        "Soil quality index",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )

    labor_days = st.number_input(
        "Labor days per ha",
        min_value=10.0,
        max_value=200.0,
        value=60.0,
        step=5.0,
    )

    distance_market = st.number_input(
        "Distance to market (km)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
    )


# =============================================================================
# PREDICTION
# =============================================================================

if st.button(
    "🔮 Predict Yield & Revenue",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Calculating prediction..."):

        # ---------------------------------------------------------------------
        # Basic values
        # ---------------------------------------------------------------------

        planting_month_num = MONTH_MAP[planting_month]

        # ---------------------------------------------------------------------
        # Weather lookup
        # ---------------------------------------------------------------------

        (
            season_avg_temp,
            season_total_rainfall,
            season_heat_days,
            weather_found,
        ) = get_season_weather(
            weather_df=weather,
            region=region,
            planting_month=planting_month_num,
            survey_year=year,
        )

        # ---------------------------------------------------------------------
        # Crop price lookup
        # ---------------------------------------------------------------------

        (
            price_per_quintal,
            price_found,
        ) = get_crop_price(
            prices_df=prices,
            region=region,
            crop_type=crop,
            survey_year=year,
        )

        # ---------------------------------------------------------------------
        # Regional mean temperature
        # ---------------------------------------------------------------------

        regional_mean_temp = get_regional_mean_temperature(
            weather_df=weather,
            region=region,
        )

        # ---------------------------------------------------------------------
        # Engineered features
        # ---------------------------------------------------------------------

        temperature_deviation = (
            season_avg_temp - regional_mean_temp
        )

        fertilizer_seed_interaction = (
            fertilizer * improved_seed
        )

        labor_per_farm_area = (
            labor_days / (farm_size + 1e-5)
        )

        fertilizer_soil_interaction = (
            fertilizer * soil_quality
        )

        month_sin = np.sin(
            2 * np.pi * planting_month_num / 12
        )

        month_cos = np.cos(
            2 * np.pi * planting_month_num / 12
        )

        # ---------------------------------------------------------------------
        # Build model input
        # ---------------------------------------------------------------------

        prediction_row = pd.DataFrame(
            [
                {
                    "region": region,
                    "crop_type": crop,
                    "survey_year": year,
                    "planting_month": planting_month,
                    "altitude_m": altitude,
                    "rainfall_mm_season": season_total_rainfall,
                    "farm_size_ha": farm_size,
                    "fertilizer_kg_per_ha": fertilizer,
                    "improved_seed_used": improved_seed,
                    "pest_disease_flag": pest_disease,
                    "soil_quality_index": soil_quality,
                    "labor_days_per_ha": labor_days,
                    "distance_to_market_km": distance_market,
                    "season_avg_temp": season_avg_temp,
                    "season_total_rainfall": season_total_rainfall,
                    "season_extreme_heat_days": season_heat_days,
                    "temp_deviation": temperature_deviation,
                    "fertilizer_x_seed": fertilizer_seed_interaction,
                    "labor_per_ha": labor_per_farm_area,
                    "fertilizer_per_ha_soil": fertilizer_soil_interaction,
                    "planting_month_num": planting_month_num,
                    "plant_month_sin": month_sin,
                    "plant_month_cos": month_cos,
                }
            ]
        )

        # ---------------------------------------------------------------------
        # Model prediction
        # ---------------------------------------------------------------------

        try:
            prediction = model.predict(prediction_row)

            if len(prediction) == 0:
                raise ValueError("The model returned no prediction.")

            predicted_yield = float(prediction[0])

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)
            st.stop()

        # Prevent displaying a negative physical yield
        predicted_yield = max(0.0, predicted_yield)

        # ---------------------------------------------------------------------
        # Revenue calculation
        #
        # 1 ton = 10 quintals
        #
        # revenue =
        # yield tons/ha × farm hectares × 10 quintals/ton
        # × price Birr/quintal
        # ---------------------------------------------------------------------

        estimated_revenue = (
            predicted_yield
            * farm_size
            * 10
            * price_per_quintal
        )


        # =============================================================================
        # RESULTS
        # =============================================================================

        st.divider()

        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Predicted Yield",
                value=f"{predicted_yield:.2f} tons/ha",
            )

        with result_col2:

            st.metric(
                label="Estimated Revenue",
                value=f"{estimated_revenue:,.0f} Birr",
            )


        # =============================================================================
        # LOOKUP INFORMATION
        # =============================================================================

        st.subheader("🌦️ Weather & Price Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric(
                "Season Avg. Temperature",
                f"{season_avg_temp:.1f} °C",
            )

        with info_col2:
            st.metric(
                "Season Rainfall",
                f"{season_total_rainfall:.0f} mm",
            )

        with info_col3:
            st.metric(
                "Extreme Heat Days",
                f"{season_heat_days:.0f}",
            )


        st.write(
            f"**Crop:** {crop}  \n"
            f"**Region:** {region}  \n"
            f"**Survey year:** {year}  \n"
            f"**Planting month:** {planting_month}  \n"
            f"**Farm size:** {farm_size:.2f} ha  \n"
            f"**Crop price:** {price_per_quintal:,.0f} Birr/quintal"
        )


        # =============================================================================
        # DATA SOURCE WARNINGS
        # =============================================================================

        if not weather_found:

            st.warning(
                "No matching weather records were found for the selected "
                "region and growing-season window. Default weather values "
                "were used for the prediction."
            )

        if not price_found:

            st.warning(
                "No matching crop price was found for the selected "
                "region, crop and survey year. A default price of "
                f"{DEFAULT_PRICE:,.0f} Birr/quintal was used."
            )


        # =============================================================================
        # REVENUE EXPLANATION
        # =============================================================================

        st.caption(
            f"Estimated revenue = {predicted_yield:.2f} tons/ha × "
            f"{farm_size:.2f} ha × 10 quintals/ton × "
            f"{price_per_quintal:,.0f} Birr/quintal."
        )


        # =============================================================================
        # YIELD CONTEXT
        # =============================================================================

        st.subheader("📈 Yield Context")

        context_data = pd.DataFrame(
            {
                "Crop / Estimate": [
                    "Predicted",
                    "Typical maize",
                    "Typical teff",
                ],
                "Yield (tons/ha)": [
                    predicted_yield,
                    3.9,
                    2.0,
                ],
            }
        ).set_index("Crop / Estimate")

        st.bar_chart(
            context_data,
            y="Yield (tons/ha)",
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Ethiopian Smallholder Crop-Yield Predictor • "
    "Machine-learning based estimation tool"
)
