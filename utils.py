"""Shared utilities for the NWS Weather Forecast app."""
import streamlit as st
import requests
from typing import Optional, Dict, Any
import traceback
import urllib3
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Disable SSL warnings for fallback method
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# NWS API base URL
NWS_API_BASE = "https://api.weather.gov"
# User agent required by NWS API
USER_AGENT = "streamlit-weather-app (contact: weather@example.com)"

# Load environment variables
load_dotenv()


class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In-memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all messages from the store."""
        self.messages = []

    def add_user_message(self, message: str) -> None:
        """Add a user message to the store."""
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the store."""
        self.messages.append(AIMessage(content=message))


def geocode_zip_code(zip_code: str) -> Optional[tuple[float, float, str]]:
    """
    Convert a ZIP code to latitude and longitude coordinates and town name.

    Tries multiple geocoding services for better reliability.

    Args:
        zip_code: US ZIP code string

    Returns:
        Tuple of (latitude, longitude, town_name) or None if geocoding fails
    """
    # Method 1: Try Zippopotam.us (ZIP code specific API)
    try:
        url = f"https://api.zippopotam.us/us/{zip_code}"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            data = response.json()
            if "places" in data and len(data["places"]) > 0:
                place = data["places"][0]
                lat = float(place.get("latitude", 0))
                lon = float(place.get("longitude", 0))
                town_name = place.get("place name", "Unknown")
                if lat != 0 and lon != 0:
                    return (lat, lon, town_name)
    except Exception:
        pass  # Fall through to next method

    # Method 2: Try Nominatim with multiple query formats
    query_formats = [
        f"{zip_code}, USA",
        f"ZIP {zip_code}, USA",
        f"United States {zip_code}"
    ]

    headers = {"User-Agent": USER_AGENT}
    url = "https://nominatim.openstreetmap.org/search"

    for query in query_formats:
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": 1,
                "countrycodes": "us"  # Restrict to US
            }

            # Try with SSL verification first
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=10,
                    verify=True
                )
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Fallback: try without SSL verification if certificate issues
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=10,
                    verify=False
                )
                response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                location = data[0]
                lat_str = location.get("lat")
                lon_str = location.get("lon")

                if lat_str and lon_str:
                    try:
                        lat = float(lat_str)
                        lon = float(lon_str)
                        # Validate coordinates are reasonable for US
                        if 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0:
                            # Extract town name from display_name or address
                            display_name = location.get("display_name", "")
                            # Try to extract city/town from display_name
                            # Format is usually: "City, State, ZIP, Country"
                            town_name = "Unknown"
                            if display_name:
                                parts = display_name.split(",")
                                if len(parts) > 0:
                                    town_name = parts[0].strip()
                            return (lat, lon, town_name)
                    except (ValueError, TypeError):
                        continue

        except requests.exceptions.RequestException:
            # Continue to next query format
            continue
        except Exception:
            # Continue to next query format
            continue

    # If all methods failed, return None
    return None


def get_grid_point(
        lat: float,
        lon: float
) -> Optional[Dict[str, Any]]:
    """
    Get NWS grid point information for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary containing grid point info or None if request fails
    """
    try:
        url = f"{NWS_API_BASE}/points/{lat},{lon}"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("Location not found in NWS database. "
                     "Please try a different ZIP code.")
        elif e.response.status_code == 503:
            st.error("NWS API is temporarily unavailable. "
                     "Please try again in a few moments.")
        else:
            st.error(f"API error: {e.response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.code(traceback.format_exc())
        return None


def get_forecast(forecast_url: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve forecast data from NWS API.

    Args:
        forecast_url: Full URL to the forecast endpoint

    Returns:
        Dictionary containing forecast data or None if request fails
    """
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(forecast_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            st.error("NWS API is temporarily unavailable. "
                     "Please try again in a few moments.")
        else:
            st.error(f"Forecast API error: {e.response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error retrieving forecast: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.code(traceback.format_exc())
        return None


def display_forecast(forecast_data: Dict[str, Any]) -> None:
    """
    Display forecast data in a formatted way.

    Args:
        forecast_data: Dictionary containing forecast data from NWS API
    """
    if not forecast_data or "properties" not in forecast_data:
        st.error("Invalid forecast data received.")
        return

    properties = forecast_data["properties"]
    periods = properties.get("periods", [])

    if not periods:
        st.warning("No forecast periods available.")
        return

    # Display location info if available
    if "relativeLocation" in properties:
        rel_loc = properties["relativeLocation"]
        location_name = rel_loc.get("properties", {}).get("city", "Unknown")
        st.subheader(f"Forecast for {location_name}")

    # Display forecast periods
    for period in periods:
        with st.expander(
                f"{period.get('name', 'Period')} - "
                f"{period.get('shortForecast', 'N/A')}",
                expanded=(period == periods[0])
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Temperature",
                    f"{period.get('temperature', 'N/A')}°F"
                )
                if "temperatureTrend" in period:
                    st.caption(f"Trend: {period['temperatureTrend']}")

            with col2:
                wind = period.get("windSpeed", "N/A")
                wind_dir = period.get("windDirection", "")
                st.metric("Wind", f"{wind_dir} {wind}".strip())

            # Detailed forecast
            detailed = period.get("detailedForecast", "")
            if detailed:
                st.write("**Details:**")
                st.write(detailed)

            # Additional info in columns
            col3, col4, col5 = st.columns(3)
            with col3:
                if "probabilityOfPrecipitation" in period:
                    pop = period["probabilityOfPrecipitation"].get("value")
                    if pop is not None:
                        st.metric("Precipitation Chance", f"{pop}%")

            with col4:
                if "dewpoint" in period:
                    dewpoint = period["dewpoint"].get("value")
                    if dewpoint is not None:
                        st.metric("Dewpoint", f"{dewpoint}°F")

            with col5:
                if "relativeHumidity" in period:
                    humidity = period["relativeHumidity"].get("value")
                    if humidity is not None:
                        st.metric("Humidity", f"{humidity}%")


def init_gemini_model() -> Optional[ChatGoogleGenerativeAI]:
    """
    Initialize LangChain Gemini model.

    Returns:
        ChatGoogleGenerativeAI model instance or None if API key is missing
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        # Use gemini-2.5-flash-lite (lightweight free tier model, version 2.5)
        # Alternative: gemini-2.5-flash or gemini-2.5-pro for more capable responses
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.7
        )
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        return None


def format_forecast_data(
        forecast_data: Dict[str, Any]
) -> str:
    """
    Format forecast data into a readable string for LLM processing.

    Args:
        forecast_data: Dictionary containing forecast data from NWS API

    Returns:
        Formatted string representation of forecast data
    """
    if not forecast_data or "properties" not in forecast_data:
        return "No forecast data available."

    properties = forecast_data["properties"]
    periods = properties.get("periods", [])

    if not periods:
        return "No forecast periods available."

    formatted = []
    for period in periods:
        period_info = f"Period: {period.get('name', 'Unknown')}\n"
        period_info += (
            f"Temperature: {period.get('temperature', 'N/A')}°F\n"
        )
        short_forecast = period.get('shortForecast', 'N/A')
        period_info += f"Short Forecast: {short_forecast}\n"
        detailed = period.get('detailedForecast', 'N/A')
        period_info += f"Detailed Forecast: {detailed}\n"

        wind = period.get("windSpeed", "N/A")
        wind_dir = period.get("windDirection", "")
        if wind_dir:
            period_info += f"Wind: {wind_dir} {wind}\n"
        else:
            period_info += f"Wind: {wind}\n"

        if "probabilityOfPrecipitation" in period:
            pop = period["probabilityOfPrecipitation"].get("value")
            if pop is not None:
                period_info += f"Precipitation Chance: {pop}%\n"

        if "relativeHumidity" in period:
            humidity = period["relativeHumidity"].get("value")
            if humidity is not None:
                period_info += f"Humidity: {humidity}%\n"

        formatted.append(period_info)

    return "\n---\n".join(formatted)


def generate_forecast_summary(
        forecast_data: Dict[str, Any],
        town_name: str,
        model: ChatGoogleGenerativeAI
) -> str:
    """
    Generate a natural language summary of the weather forecast using Gemini.

    Args:
        forecast_data: Dictionary containing forecast data from NWS API
        town_name: Name of the town/city
        model: Initialized Gemini model

    Returns:
        English language summary of the forecast
    """
    try:
        formatted_forecast = format_forecast_data(forecast_data)

        prompt = (
            f"Generate a natural language summary of this weather "
            f"forecast for {town_name}.\n\n"
            f"Forecast Data:\n{formatted_forecast}\n\n"
            f"Please provide a clear, concise summary in plain English "
            f"that highlights the key weather patterns, temperatures, "
            f"and conditions over the forecast period. Make it "
            f"conversational and easy to understand."
        )

        system_msg = (
            "You are a helpful weather assistant that provides clear, "
            "natural language summaries of weather forecasts."
        )
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt)
        ]

        response = model.invoke(messages)
        # Handle different response types
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content
            elif hasattr(content, '__str__'):
                return str(content)
            else:
                return str(response)
        else:
            return str(response)
    except Exception as e:
        error_str = str(e)
        # Check for quota/resource exhausted errors
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
            st.warning(
                "⚠️ **API Quota Exceeded:** The free tier limit for Google Gemini API "
                "has been reached (20 requests per day). The forecast summary feature "
                "is temporarily unavailable. Please try again tomorrow or upgrade "
                "your API plan.\n\n"
                "You can still view the detailed forecast above."
            )
        else:
            error_trace = traceback.format_exc()
            error_line = (
                error_trace.split('\n')[-2]
                if error_trace else "Unknown"
            )
            st.error(
                f"Error generating forecast summary "
                f"(line {error_line}): {str(e)}"
            )
        return (
            "Unable to generate forecast summary. "
            "Please refer to the detailed forecast above."
        )


def get_or_create_chat_history(
        conversation_id: str
) -> InMemoryChatMessageHistory:
    """
    Get or create a chat message history for a conversation ID.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        InMemoryChatMessageHistory instance for the conversation
    """
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if conversation_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[conversation_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.chat_histories[conversation_id]


def answer_weather_question(
        question: str,
        forecast_data: Dict[str, Any],
        town_name: str,
        conversation_id: str,
        model: ChatGoogleGenerativeAI
) -> str:
    """
    Answer a weather-related question using Gemini with forecast context.

    Uses LangChain's chat message history to persist conversation context
    between calls using a conversation ID.

    Args:
        question: User's question
        forecast_data: Dictionary containing forecast data from NWS API
        town_name: Name of the town/city
        conversation_id: Unique identifier for the conversation session
        model: Initialized Gemini model

    Returns:
        Answer to the question
    """
    try:
        # Get or create chat history for this conversation
        chat_history = get_or_create_chat_history(conversation_id)

        formatted_forecast = format_forecast_data(forecast_data)

        # Build system message with forecast context
        system_msg = (
            f"You are a helpful weather assistant. Answer questions based "
            f"on the provided weather forecast data for {town_name}. "
            f"Be concise and accurate.\n\n"
            f"Current weather forecast for {town_name}:\n"
            f"{formatted_forecast}"
        )

        # Get previous messages from history
        previous_messages = chat_history.messages

        # Build message list: system message, history, and new question
        messages = [SystemMessage(content=system_msg)]
        messages.extend(previous_messages)
        messages.append(HumanMessage(content=question))

        # Create config with conversation context and metadata
        config = RunnableConfig(
            tags=["weather-qa", f"location-{town_name}"],
            metadata={
                "conversation_id": conversation_id,
                "town_name": town_name,
                "forecast_available": True
            }
        )
        
        # Invoke model with full conversation context and config
        response = model.invoke(
            messages,
            config=config
        )

        # Extract response content
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                answer = content
            elif hasattr(content, '__str__'):
                answer = str(content)
            else:
                answer = str(response)
        else:
            answer = str(response)

        # Add question and answer to chat history
        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)

        return answer
    except Exception as e:
        error_str = str(e)
        # Check for quota/resource exhausted errors
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
            st.warning(
                "⚠️ **API Quota Exceeded:** The free tier limit for Google Gemini API "
                "has been reached (20 requests per day). The Q&A feature is temporarily "
                "unavailable. Please try again tomorrow or upgrade your API plan."
            )
            return (
                "I'm sorry, the AI assistant is temporarily unavailable due to API "
                "quota limits. Please try again tomorrow or check the detailed forecast "
                "above for weather information."
            )
        else:
            error_trace = traceback.format_exc()
            error_line = (
                error_trace.split('\n')[-2]
                if error_trace else "Unknown"
            )
            st.error(
                f"Error answering question (line {error_line}): {str(e)}"
            )
            return (
                "I'm sorry, I encountered an error while processing your "
                "question. Please try again."
            )

