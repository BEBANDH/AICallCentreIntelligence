import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Check for required API keys
if not api_key or not google_api_key:
    st.error("Please set valid OpenWeatherMap and Google API keys in your environment variables.")
    st.stop()

# Initialize the LLM with Google Generative AI
llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)

# Define the prompt template for the chatbot
prompt = PromptTemplate(
    input_variables=["weather_details", "user_question"],
    template=(
        "You are an expert event and outdoor activity planner. "
        "Based on the following weather details for a specific location:\n\n"
        "{weather_details}\n\n"
        "Answer the user's question: \"{user_question}\""
        "\n\n"
        "Your response should include:"
        "\n* Suggestions for indoor or outdoor venues based on the weather."
        "\n* Weather-based recommendations for outdoor activities (hiking, biking, camping)."
        "\n* Consider factors like temperature, humidity, wind, rain, and cloud cover."
        "\n* Provide concise and informative answers."
        "\n* Include names of local places , timings of local places and the name of 3 to 4 local food items."
    )
)

# Create the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Streamlit UI
st.set_page_config(page_title="Weather-Based Event & Activity Planner", layout="centered")
st.title("Weather-Based Event & Activity Planner")

# Function to get weather data using OpenWeatherMap API
def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Initialize session state variables
if "weather_details" not in st.session_state:
    st.session_state.weather_details = None
if "place" not in st.session_state:
    st.session_state.place = None

# Input city and fetch weather data
with st.form("weather_form"):
    place_input = st.text_input("Enter the place:", value=st.session_state.place or "")
    submit_weather = st.form_submit_button("Get Weather")
    
    if submit_weather and place_input:
        st.session_state.place = place_input
        weather_data = get_weather_data(place_input)
        
        if "error" not in weather_data:
            st.session_state.weather_details = (
                f"In {place_input}, {weather_data.get('sys', {}).get('country', 'Unknown')}, "
                f"the current weather is as follows:\n"
                f"Detailed status: {weather_data['weather'][0]['description']}\n"
                f"Wind speed: {weather_data['wind']['speed']} m/s, direction: {weather_data['wind']['deg']}°\n"
                f"Humidity: {weather_data['main']['humidity']}%\n"
                f"Temperature:\n"
                f"  - Current: {weather_data['main']['temp']}°C\n"
                f"  - High: {weather_data['main']['temp_max']}°C\n"
                f"  - Low: {weather_data['main']['temp_min']}°C\n"
                f"Cloud cover: {weather_data['clouds']['all']}%\n"
            )
        else:
            st.error(weather_data["error"])

# Display weather details if available
if st.session_state.weather_details:
    st.subheader("Weather Details")
    st.write(st.session_state.weather_details)
    
    # Input question and get recommendations
    with st.form("question_form"):
        user_question = st.text_input("Ask your question about events or activities:")
        submit_question = st.form_submit_button("Get Recommendations")
        
        if submit_question and user_question:
            try:
                response = chain.run(
                    weather_details=st.session_state.weather_details,
                    user_question=user_question
                )
                st.subheader("Recommendations")
                st.write(response)
            except Exception as e:
                st.error(f"Error running the LLM chain: {e}")
