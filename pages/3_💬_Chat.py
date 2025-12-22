"""Page for interactive weather Q&A chat."""
import streamlit as st
import traceback
import uuid
from utils import (
    answer_weather_question,
    get_or_create_conversation_memory,
    init_gemini_model
)
from langchain_core.messages import HumanMessage, AIMessage

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

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

st.title("💬 Ask About the Weather")

# Check if we have forecast data
if st.session_state.current_forecast_data is None:
    st.warning(
        "⚠️ **No forecast data available.** Please go to the 'Enter ZIP Code' "
        "page to enter a ZIP code, then visit the 'Forecast' page to load "
        "the weather forecast."
    )
    st.info(
        "Once you have a forecast loaded, you can ask questions about it here."
    )
elif st.session_state.gemini_model is None:
    st.warning(
        "⚠️ **Google Gemini API key not found.** Please create a .env file "
        "with GOOGLE_API_KEY to enable the Q&A feature."
    )
else:
    # We have forecast data and Gemini model
    town_name = st.session_state.current_town_name or "the location"
    st.info(f"📍 **Location:** {town_name}")

    # Get conversation memory for this conversation
    memory = get_or_create_conversation_memory(
        st.session_state.conversation_id
    )
    # Load messages from ConversationBufferMemory
    history_dict = memory.load_memory_variables({})
    messages = history_dict.get("history", [])

    # Display conversation history
    if messages:
        st.markdown("### Conversation History")
        
        # Group messages into Q&A pairs
        pairs = []
        current_q = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                current_q = msg.content
            elif isinstance(msg, AIMessage):
                if current_q:
                    pairs.append((current_q, msg.content))
                    current_q = None
        
        # Display pairs - most recent first
        if pairs:
            # Show most recent pair at top
            if len(pairs) > 0:
                q, a = pairs[-1]
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                
            # Show older pairs in expanders
            if len(pairs) > 1:
                st.divider()
                st.markdown("**Previous Questions:**")
                for i, (q, a) in enumerate(reversed(pairs[:-1]), 1):
                    q_text = q[:50] + "..." if len(q) > 50 else q
                    with st.expander(
                        f"Q{i+1}: {q_text}",
                        expanded=False
                    ):
                        st.markdown(f"**Question:** {q}")
                        st.markdown(f"**Answer:** {a}")
            
            st.divider()
        elif len(messages) > 0:
            # Fallback: show messages directly if pairing didn't work
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"**Q:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.markdown(f"**A:** {msg.content}")
            st.divider()

    # Question input
    placeholder_text = (
        "e.g., Will it rain tomorrow? "
        "What's the best day for outdoor activities? "
        "How warm will it be this week?"
    )
    
    user_question = st.text_input(
        "What would you like to know?",
        placeholder=placeholder_text,
        key="user_question_input"
    )
    
    col1, col2 = st.columns([1, 10])
    with col1:
        ask_button = st.button("Ask", type="primary", key="ask_button")
    
    # Process question if button clicked
    if ask_button and user_question and user_question.strip():
        question_text = user_question.strip()
        with st.spinner("Thinking..."):
            try:
                answer = answer_weather_question(
                    question_text,
                    st.session_state.current_forecast_data,
                    st.session_state.current_town_name,
                    st.session_state.conversation_id,
                    st.session_state.gemini_model
                )

                if answer:
                    # Display the answer immediately
                    st.success("Response received!")
                    st.markdown("### Latest Response")
                    st.markdown(f"**Q:** {question_text}")
                    st.markdown(f"**A:** {answer}")
                    st.divider()
                    # Rerun to refresh the page and show updated history
                    # The input will be cleared automatically on rerun
                    st.rerun()
                else:
                    st.error(
                        "No response received. Please try again."
                    )
            except Exception as e:
                error_trace = traceback.format_exc()
                st.error(
                    f"Error processing question: {str(e)}\n\n"
                    f"Details: {error_trace}"
                )

    # Show example questions
    st.divider()
    st.markdown("### Example Questions")
    st.markdown("""
    - Will it rain tomorrow?
    - What's the best day for outdoor activities?
    - How warm will it be this week?
    - Is there any chance of snow?
    - What's the wind speed going to be?
    - Should I bring an umbrella?
    """)

