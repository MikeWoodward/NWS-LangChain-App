"""Page for displaying weather forecast and AI summary."""
import streamlit as st
import uuid
from utils import (
    get_grid_point,
    get_forecast,
    display_forecast,
    init_gemini_model,
    generate_forecast_summary
)

# Initialize session state if not already initialized
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

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

st.title("🌤️ Weather Forecast")

# Show warning if Gemini is not available
if st.session_state.gemini_model is None:
    st.warning(
        "⚠️ Google Gemini API key not found. Forecast summaries will "
        "not be available. Please create a .env file with "
        "GOOGLE_API_KEY to enable AI features."
    )

# Check if we have location data
if st.session_state.current_town_name is None:
    st.warning(
        "📍 **No location selected.** Please go to the 'Enter ZIP Code' "
        "page first to enter a ZIP code and find a location."
    )
    st.info("Once you've entered a ZIP code, return here to see the forecast.")
else:
    town_name = st.session_state.current_town_name
    st.info(f"📍 **Location:** {town_name}")

    # Check if we have coordinates
    if st.session_state.current_coords is None:
        st.error("Coordinates not available. Please enter a ZIP code first.")
    else:
        lat, lon = st.session_state.current_coords

        # Check if forecast data is already loaded
        if st.session_state.current_forecast_data is None:
            with st.spinner("Loading weather forecast..."):
                # Get grid point
                grid_data = get_grid_point(lat, lon)

                if grid_data and "properties" in grid_data:
                    props = grid_data["properties"]
                    forecast_url = props.get("forecast")

                    if forecast_url:
                        # Get forecast
                        forecast_data = get_forecast(forecast_url)

                        if forecast_data:
                            st.session_state.current_forecast_data = (
                                forecast_data
                            )
                            # Clear summary when new forecast is loaded
                            st.session_state.forecast_summary = None
                            # Reset conversation ID when new forecast is loaded
                            st.session_state.conversation_id = str(uuid.uuid4())
                            # Clear chat histories
                            if "chat_histories" in st.session_state:
                                st.session_state.chat_histories = {}
                            st.rerun()
                        else:
                            st.error("Failed to retrieve forecast data.")
                    else:
                        st.error(
                            "Forecast URL not found in grid point data."
                        )
                else:
                    st.error("Failed to get grid point information.")

        # Display forecast if available
        if st.session_state.current_forecast_data:
            forecast_data = st.session_state.current_forecast_data

            # Display forecast
            display_forecast(forecast_data)

            # Generate and display summary if Gemini is available
            if st.session_state.gemini_model:
                st.divider()
                st.subheader("📝 Forecast Summary")

                if st.session_state.forecast_summary is None:
                    with st.spinner("Generating forecast summary..."):
                        summary = generate_forecast_summary(
                            forecast_data,
                            town_name,
                            st.session_state.gemini_model
                        )
                        st.session_state.forecast_summary = summary
                        st.markdown(summary)
                else:
                    st.markdown(st.session_state.forecast_summary)
            else:
                st.info(
                    "💡 **Tip:** Set up your Google API key to enable "
                    "AI-generated forecast summaries."
                )

            st.divider()
            st.info(
                "💬 **Want to ask questions about this forecast?** "
                "Visit the **Chat** page to interact with the weather "
                "assistant."
            )

