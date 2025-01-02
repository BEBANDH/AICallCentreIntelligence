import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from gtts import gTTS
from io import BytesIO
from streamlit_chat import message

# Load environment variables
load_dotenv()

# Initialize the LLM with Google Generative AI
llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Define the prompt template for the chatbot
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input", "product_problem"],
    template=(
        "You are a highly skilled customer care representative dedicated to resolving users' product-related issues. "
        "The user has described the following problem with their product: {product_problem}. "
        "Provide clear and practical solutions to address their concerns, including troubleshooting steps, potential timelines for resolution, warranty information, and any additional support they might need. "
        "Feel free to invent plausible details where necessary to offer a seamless customer service experience. "
        "Engage with the user in a polite, professional, and conversational tone, ensuring their satisfaction. "
        "Ensure that you give small answers in short points , that is not too long to read "
        "\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User: {user_input}\n\n"
        "Agent:"
    )
)

# Create the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Streamlit UI
st.set_page_config(page_title="Customer Care Chatbot", layout="centered")
st.title("Customer Care Chatbot")
st.write(
    """
    **Disclaimer**  
    We appreciate your engagement! Please note, this bot is designed to assist with product-related issues.  
    Type your queries below to get started.
    """
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "product_problem" not in st.session_state:
    st.session_state["product_problem"] = ""

# Step 1: Input the product and problem
if not st.session_state["product_problem"]:
    st.subheader("Step 1: Describe Your Problem")
    product_problem_input = st.text_area("What issue are you facing with your product?")
    if st.button("Submit Problem"):
        if product_problem_input.strip():
            st.session_state["product_problem"] = product_problem_input.strip()
            st.success(f"Problem noted: {st.session_state['product_problem']}")
        else:
            st.warning("Please describe your problem.")
else:
    st.subheader(f"Problem: {st.session_state['product_problem']}")

    # Step 2: Chat UI
    st.subheader("Chat with the Support Bot")

    # Display chat history dynamically
    for i,chat in enumerate (st.session_state["chat_history"]):
        message(chat["user"], is_user=True, key=f"user_{i}")  # User's message
        message(chat["agent"], is_user=False, key=f"agent_{i}")  # Bot's response

    # Input box at the bottom
    user_input = st.text_input("Type your message:", key="chat_input", label_visibility="collapsed")

    if st.button("Send Message"):
        if user_input.strip():
            if user_input.lower() == "quit":
                st.success("Conversation ended. Thank you for reaching out!")
                st.session_state["chat_history"].append({"user": user_input, "agent": "Goodbye!"})
                memory.clear()
            else:
                # Generate response from the LLM
                try:
                    chat_history_text = "\n".join(
                        [f"User: {item['user']}\nAgent: {item['agent']}" for item in st.session_state["chat_history"]]
                    )
                    response = chain.run(
                        chat_history=chat_history_text,
                        user_input=user_input,
                        product_problem=st.session_state["product_problem"]
                    )
                    st.session_state["chat_history"].append({"user": user_input, "agent": response.strip()})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a message.")

    # Reset chat
    if st.button("Reset Chat"):
        st.session_state["product_problem"] = ""
        st.session_state["chat_history"] = []
        memory.clear()
        st.success("Chat reset successfully!")
