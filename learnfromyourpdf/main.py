from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class KnowledgeRetriever:
    def __init__(self, file_path, model) -> None:
        self.file_path = file_path
        self.loader = None
        self.chunks = None
        self.vectorstore = None
        self.embedding = None
        self.retriever = None
        self.retrieved_docs = None
        self.llm = ChatGroq(model=model)

    def create_retriever(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()
        self.chunks = pages
        vectorstore = Chroma.from_documents(
            documents=pages,
            embedding=OpenAIEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()

    def create_vectorstore(self):
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=OpenAIEmbeddings(),
        )

    # def create_retriever(self):
    #     self.retriever = self.vectorstore.as_retriever()

    def retrieve(self, query):
        retrieved_docs = self.retriever.invoke(query)
        self.retrieved_docs = retrieved_docs
        return retrieved_docs

    def prompt_template(self, template):
        prompt = PromptTemplate.from_template(template)
        return prompt

    def start_rag_chain(self, question):
        # First load the pdf and split it into chunks

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, use LLM knowledge to answer the question.
        {context}

        Question: {question}

        Helpful Answer:"""

        prompt = self.prompt_template(template=template)
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        # responses = []
        return rag_chain.stream(question)
        # responses.append(chunk)

        # return responses
