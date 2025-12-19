# NWS Weather Forecast App

A Streamlit-based weather forecast application that provides detailed weather information using the National Weather Service (NWS) API, enhanced with AI-powered summaries and interactive Q&A capabilities powered by Google Gemini.

## Overview

This application allows users to:
- Enter a US ZIP code to find the nearest town and coordinates
- View detailed 7-day weather forecasts from the National Weather Service
- Get AI-generated natural language summaries of weather forecasts
- Ask questions about the weather forecast in an interactive chat interface

## Features

### 📍 Location Lookup
- Enter any 5-digit US ZIP code
- Automatic geocoding using multiple fallback services (Zippopotam.us and Nominatim)
- Displays town name and coordinates
- Validates ZIP code format and location

### 🌤️ Detailed Forecast
- 7-day weather forecast from the National Weather Service
- Temperature, wind speed and direction, precipitation probability
- Dewpoint and relative humidity information
- Detailed forecast descriptions
- AI-generated natural language summaries (requires Google API key)

### 💬 Interactive Chat
- Ask questions about the weather forecast
- AI-powered answers using Google Gemini 2.5 Flash-Lite
- Conversation history maintained across questions
- Context-aware responses based on the current forecast

## Architecture

### Application Structure

```
NWS LangChain app/
├── app.py                          # Main splash page and session state initialization
├── utils.py                        # Core utility functions and API interactions
├── requirements.txt                # Python dependencies
├── pages/
│   ├── 1_📍_Enter_ZIP_Code.py      # ZIP code entry and geocoding page
│   ├── 2_🌤️_Forecast.py           # Weather forecast display page
│   └── 3_💬_Chat.py                # Interactive Q&A chat page
└── venv/                           # Virtual environment (not in git)
```

### Key Components

#### `app.py`
- Main entry point for the Streamlit application
- Initializes session state variables:
  - `gemini_model`: Google Gemini model instance
  - `current_town_name`: Currently selected town/city
  - `current_coords`: Latitude and longitude coordinates
  - `current_forecast_data`: Cached forecast data from NWS API
  - `forecast_summary`: AI-generated forecast summary
  - `conversation_history`: Chat conversation history
  - `conversation_id`: Unique identifier for chat sessions
- Displays welcome page with feature overview

#### `utils.py`
Core utility functions:

**Geocoding:**
- `geocode_zip_code()`: Converts ZIP codes to coordinates using multiple fallback services

**NWS API Integration:**
- `get_grid_point()`: Retrieves NWS grid point information for coordinates
- `get_forecast()`: Fetches forecast data from NWS API
- `display_forecast()`: Formats and displays forecast data in Streamlit UI

**AI Integration:**
- `init_gemini_model()`: Initializes Google Gemini model via LangChain
- `format_forecast_data()`: Formats forecast data for LLM processing
- `generate_forecast_summary()`: Creates natural language forecast summaries
- `answer_weather_question()`: Answers user questions with forecast context

**Chat History:**
- `InMemoryChatMessageHistory`: In-memory implementation of chat message history
- `get_or_create_chat_history()`: Manages conversation history per session

#### `pages/1_📍_Enter_ZIP_Code.py`
- ZIP code input and validation
- Geocoding using `geocode_zip_code()` utility
- Displays location information and coordinates
- Stores location data in session state

#### `pages/2_🌤️_Forecast.py`
- Displays weather forecast for selected location
- Fetches forecast data from NWS API if not cached
- Shows detailed forecast periods with expandable sections
- Generates AI summary if Gemini model is available
- Resets conversation ID when new forecast is loaded

#### `pages/3_💬_Chat.py`
- Interactive Q&A interface for weather questions
- Uses conversation history for context-aware responses
- Displays conversation history with expandable previous questions
- Validates that forecast data and Gemini model are available

## Dependencies

### Required Packages
- `streamlit>=1.28.0`: Web application framework
- `requests>=2.31.0`: HTTP library for API calls
- `langchain>=0.1.0`: LLM framework for AI integration
- `langchain-google-genai>=1.0.0`: Google Gemini integration for LangChain
- `python-dotenv>=1.0.0`: Environment variable management

### External APIs
- **National Weather Service API**: Free, public weather data API
  - Base URL: `https://api.weather.gov`
  - Requires User-Agent header
  - No API key required
  
- **Google Gemini API**: AI model for summaries and Q&A
  - Model: `gemini-2.5-flash-lite` (free tier)
  - Requires API key in `.env` file
  - Alternative: `gemini-2.5-pro` for more capable responses

- **Geocoding Services** (fallback chain):
  1. Zippopotam.us: ZIP code-specific API
  2. Nominatim (OpenStreetMap): General geocoding service

## Setup Instructions

### 1. Clone or Navigate to Project Directory
```bash
cd "NWS LangChain app"
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

**Note:** The application will work without the Google API key, but AI features (forecast summaries and Q&A) will be disabled.

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Step 1: Enter ZIP Code
1. Navigate to the "📍 Enter ZIP Code" page
2. Enter a valid 5-digit US ZIP code (e.g., `10001`)
3. The app will find the location and display town name and coordinates

### Step 2: View Forecast
1. Go to the "🌤️ Forecast" page
2. The app automatically fetches the 7-day forecast from NWS
3. View detailed forecast information in expandable sections
4. If Gemini is configured, view the AI-generated summary

### Step 3: Ask Questions (Optional)
1. Visit the "💬 Chat" page
2. Ask questions about the weather forecast
3. View conversation history and ask follow-up questions

## Technical Details

### Session State Management
The application uses Streamlit's session state to maintain:
- Location information across pages
- Cached forecast data to avoid redundant API calls
- Conversation history for context-aware Q&A
- Unique conversation IDs for session management

### Error Handling
- Comprehensive error handling for API failures
- Graceful fallbacks for geocoding services
- User-friendly error messages
- Exception traceback logging with line numbers

### API Rate Limiting
- NWS API: No official rate limits, but be respectful
- Nominatim: Requires User-Agent header, rate-limited
- Google Gemini: Subject to API quota limits

### Geocoding Strategy
The app uses a multi-service fallback approach:
1. **Primary**: Zippopotam.us (ZIP-specific, fast)
2. **Fallback**: Nominatim with multiple query formats
3. **Validation**: Coordinate bounds checking for US locations

### AI Model Configuration
- **Model**: `gemini-2.5-flash-lite` (free tier, lightweight and fast responses)
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Context**: Full forecast data included in system message
- **History**: Maintained per conversation ID

## Code Style

The codebase follows:
- **PEP 8** style guidelines
- **Type hints** for function parameters and return values
- **Docstrings** for all functions and classes
- **f-strings** for string formatting
- **Keyword arguments** where appropriate
- **Meaningful variable names**
- **Error handling** with line number reporting

## Limitations

1. **US ZIP Codes Only**: The application currently supports only US ZIP codes
2. **NWS Coverage**: Weather data is only available for locations within NWS coverage areas
3. **API Dependencies**: Requires internet connection and API availability
4. **Gemini API Key**: AI features require a valid Google API key

## Future Enhancements

Potential improvements:
- Support for international locations
- Historical weather data
- Weather alerts and warnings
- Multiple location comparison
- Export forecast data
- Customizable forecast periods
- Weather maps integration

## License

This project uses:
- **NWS API**: Public domain data from NOAA
- **Streamlit**: Apache License 2.0
- **LangChain**: MIT License
- **Google Gemini**: Subject to Google's Terms of Service

## Contributing

When contributing:
1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings
4. Handle errors gracefully with line number reporting
5. Test with various ZIP codes and edge cases

## Support

For issues or questions:
- Check that your ZIP code is valid and in the US
- Verify your Google API key is correctly configured
- Ensure you have an active internet connection
- Check that the NWS API is accessible

## Acknowledgments

- **National Weather Service** for providing free, public weather data
- **Google** for Gemini AI capabilities
- **Streamlit** for the web framework
- **LangChain** for LLM integration tools

