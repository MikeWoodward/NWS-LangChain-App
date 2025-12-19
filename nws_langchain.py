"""Splash page for NWS Weather Forecast app."""
import streamlit as st
from utils import init_gemini_model

st.set_page_config(
    page_title="NWS Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# Initialize session state for all pages
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = init_gemini_model()

if "current_town_name" not in st.session_state:
    st.session_state.current_town_name = None

if "current_coords" not in st.session_state:
    st.session_state.current_coords = None

if "current_forecast_data" not in st.session_state:
    st.session_state.current_forecast_data = None

if "forecast_summary" not in st.session_state:
    st.session_state.forecast_summary = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "conversation_id" not in st.session_state:
    import uuid
    st.session_state.conversation_id = str(uuid.uuid4())

st.title("🌤️ National Weather Service Forecast")
st.markdown("---")

st.header("Welcome!")
st.markdown(
    """
    This app provides weather forecasts using data from the National Weather 
    Service (NWS) API, enhanced with AI-powered summaries and interactive 
    Q&A capabilities.
    """
)

st.markdown("### Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📍 Location Lookup**
    - Enter a US ZIP code
    - Find the nearest town
    - Get accurate coordinates
    """)

with col2:
    st.markdown("""
    **🌤️ Detailed Forecast**
    - 7-day weather forecast
    - Temperature, wind, precipitation
    - AI-generated summaries
    """)

with col3:
    st.markdown("""
    **💬 Interactive Chat**
    - Ask questions about the weather
    - Get AI-powered answers
    - Conversation history
    """)

st.markdown("---")

st.markdown("### How to Use")
st.markdown("""
1. **Enter ZIP Code** - Go to the "Enter ZIP Code" page and enter a 
   5-digit US ZIP code
2. **View Forecast** - Check the "Forecast" page to see detailed weather 
   information and AI-generated summaries
3. **Ask Questions** - Visit the "Chat" page to ask questions about 
   the weather forecast
""")

st.markdown("---")

st.markdown("### About")
st.markdown("""
This application uses:
- **National Weather Service API** for official weather data
- **Google Gemini 2.5 Flash-Lite** for AI-powered summaries and Q&A
- **Streamlit** for the user interface

All weather data is provided by the National Weather Service, a division 
of NOAA.
""")

st.markdown("---")
st.info(
    "💡 **Tip:** Make sure you have a valid Google API key in your `.env` "
    "file to enable AI features (forecast summaries and Q&A)."
)
