import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

memory = ConversationBufferMemory()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant that answers strictly based on the given passage. "
        "If the question is not related to the passage, respond with: "
        "'I have no information regarding this.'\n\n"
        "Passage:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

st.title("Passage-QA Chatbot with Google")
st.header("Enter Your Passage Below")

passage_input = st.text_area("Enter your passage here:")
if st.button("Submit Passage"):
    memory.save_context(
        {"input": "User provided a passage."},
        {"output": passage_input}
    )
    st.success("Passage stored in memory!")

st.header("Ask Questions About the Passage")
question_input = st.text_input("Enter your question:")
if st.button("Send Question"):
    if not passage_input.strip():
        st.warning("Please enter and submit a passage first!")
    else:
        try:
            response = chain.run(context=passage_input, question=question_input)
            st.text_area("Answer:", value=response.strip(), height=100)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.button("Reset Memory"):
    memory.clear()
    st.success("Memory reset successfully!")
