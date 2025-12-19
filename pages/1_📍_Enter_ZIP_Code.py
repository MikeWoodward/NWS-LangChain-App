"""Page for entering ZIP code and finding nearest town."""
import streamlit as st
from utils import geocode_zip_code

# Initialize session state if not already initialized
if "current_town_name" not in st.session_state:
    st.session_state.current_town_name = None

if "current_coords" not in st.session_state:
    st.session_state.current_coords = None

st.title("📍 Enter ZIP Code")
st.markdown("Enter a US ZIP code to find the nearest town and get weather "
            "forecast information.")

# ZIP code input
zip_code = st.text_input(
    "Enter ZIP Code",
    placeholder="e.g., 10001",
    help="Enter a 5-digit US ZIP code",
    key="zip_code_input"
)

if zip_code:
    # Validate ZIP code format (basic check)
    zip_code = zip_code.strip()
    if not zip_code.isdigit() or len(zip_code) != 5:
        st.error("Please enter a valid 5-digit US ZIP code.")
    else:
        with st.spinner("Finding location..."):
            # Geocode ZIP to coordinates
            coords = geocode_zip_code(zip_code)

            if coords:
                lat, lon, town_name = coords
                st.session_state.current_town_name = town_name
                st.session_state.current_coords = (lat, lon)

                st.success(f"📍 **Location Found:** {town_name}")
                st.info(
                    f"**Coordinates:** {lat:.4f}°N, {lon:.4f}°W\n\n"
                    f"Navigate to the **Forecast** page to see the weather "
                    f"forecast for {town_name}."
                )
            else:
                st.error(
                    f"Could not find coordinates for ZIP code {zip_code}. "
                    "Please verify the ZIP code is correct and try again. "
                    "Note: Only US ZIP codes are supported."
                )
                st.info(
                    "💡 **Tip:** Make sure you're entering a valid "
                    "5-digit US ZIP code. If the issue persists, "
                    "the geocoding service may be temporarily unavailable."
                )

# Show current location if available
if st.session_state.current_town_name:
    st.divider()
    st.markdown("### Current Location")
    st.markdown(f"**Town:** {st.session_state.current_town_name}")
    if st.session_state.current_coords:
        lat, lon = st.session_state.current_coords
        st.markdown(f"**Coordinates:** {lat:.4f}°N, {lon:.4f}°W")

