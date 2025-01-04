import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

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
    input_variables=["chat_history", "user_input", "product"],
    template=(
        "You are a call center agent tasked with selling the following product: {product}. "
        "Invent details such as specifications, price, package tiers, colors, and other relevant features if the user asks. "
        "Engage with the user in a conversational style to provide information, answer questions, and persuade them to buy the product. "
        "If the user seems convinced, conclude the chat appropriately. If the user says 'quit', end the conversation."
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
st.title("Sales Chatbot")
st.header("Call Center Agent Simulation")

# Step 1: Input the product to sell
if "product" not in st.session_state:
    st.session_state["product"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if not st.session_state["product"]:
    st.subheader("Step 1: Enter the Product")
    product_input = st.text_input("What product do you want to sell?")
    if st.button("Start Selling"):
        if product_input.strip():
            st.session_state["product"] = product_input.strip()
            st.success(f"Product set to: {st.session_state['product']}")
        else:
            st.warning("Please enter a product.")
else:
    st.subheader(f"Product: {st.session_state['product']}")

    # Step 2: Chat UI
    st.subheader("Chat with the Sales Bot")

    # Display chat history dynamically
    st.divider()
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Agent:** {chat['agent']}")

    # Input box placed at the bottom
    user_input = st.text_input("Your message:", key="chat_input")

    if st.button("Send Message"):
        if user_input.strip():
            if user_input.lower() == "quit":
                st.success("Conversation ended. Thank you!")
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
                        product=st.session_state["product"]
                    )
                    st.session_state["chat_history"].append({"user": user_input, "agent": response.strip()})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a message.")

    # Reset chat
    if st.button("Reset Chat"):
        st.session_state["product"] = ""
        st.session_state["chat_history"] = []
        memory.clear()
        st.success("Chat reset successfully!")
