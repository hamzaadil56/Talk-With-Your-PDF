import streamlit as st
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
from learnfromyourpdf.main import KnowledgeRetriever
import tempfile
import pathlib
import os

knowledge_retriever = None

st.title("Talk With Your PDF")

with st.container():
    uploaded_file = st.file_uploader(
        'Upload a file from which you want to talk')

temp_dir = tempfile.TemporaryDirectory()

uploaded_file_name = "File_provided"
uploaded_file_path = pathlib.Path(temp_dir.name) / uploaded_file_name

if uploaded_file is not None:
    with open(uploaded_file_path, 'wb') as output_temporary_file:
        output_temporary_file.write(uploaded_file.read())
    knowledge_retriever = KnowledgeRetriever(
        uploaded_file_path, 'llama3-70b-8192')
    knowledge_retriever.create_retriever()
    st.write("File uploaded successfully!")


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Talk with me to extract information from your pdf!"),
    ]


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(
            knowledge_retriever.start_rag_chain(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))


# Initialize the session state to keep track of conversation history
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# Set page layout


# col1, col2 = st.columns([9, 1], vertical_alignment='bottom')
# Define a function for the chatbot's response


# Display the chat history

# Accept user input

# user_input = col1.text_input("You: ", key="user_input",
#                              placeholder="Type your message...")


# if col2.button("Send"):
#     if uploaded_file is None:
#         # Add user message to chat history
#         st.error('Please upload file to talk with your pdf.', icon="🚨")

#     if user_input:
#         # Add user message to chat history
#         st.session_state["messages"].append(
#             {"role": "user", "content": user_input})
#         if knowledge_retriever is not None:
#             st.write(knowledge_retriever.chunks)
#             knowledge_retriever.create_vectorstore()
#             st.write(knowledge_retriever.vectorstore)
#             knowledge_retriever.create_retriever()
#             st.write(knowledge_retriever.retriever)
#             responses = knowledge_retriever.start_rag_chain(
#                 question=user_input)

#             st.session_state["messages"].append(
#                 {"role": "assistant", "content": ''.join(responses)})

# for msg in st.session_state["messages"]:
#     st.write(msg['content'])
