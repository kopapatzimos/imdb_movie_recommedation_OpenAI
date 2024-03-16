import pandas as pd 
import re
import tiktoken
import os
import openai


from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from  langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
import dill
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



def normalize_text(text):
    # Remove special characters, punctuation, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def read_imdb_csv(csv_file: str):
    df = pd.read_csv('imdb_movies.csv')
    df.drop_duplicates(subset="orig_title",inplace=True)
    # Remove records with any missing values
    df = df.dropna()
    
    # Apply text normalization to 'overview' column
    df['overview'] = df['overview'].apply(normalize_text)

    # Combine Columns
    df['combined_details'] = df.apply(lambda row: f"Title: {row['names']}. Overview: {row['overview']} Genres: {row['genre']} Relese_date:{row['date_x']} Languages:{row['orig_lang']}", axis=1)
    #Save processed dataset - combined_info for Langchain
    df[['combined_details']].to_csv('imdb_movies_updated.csv', index=False)
    
    final_df = pd.read_csv('imdb_movies_updated.csv')
    return final_df




# Initialize required components
df = read_imdb_csv(csv_file='imdb_movies.csv')
OPENAI_API_KEY = ''
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
loader = CSVLoader(file_path="imdb_movies_updated.csv")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
qa = RetrievalQA.from_chain_type(llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=True)

# Define function to process user input and generate response
def process_query(query):
    result = qa({"query": query})
    return result['result']

# Example conversation loop
while True:
    user_input = input("You: ")
    response = process_query(user_input)
    print("Bot:", response)


# from langchain.prompts import PromptTemplate

# template = """You are a movie recommender system that help users to find anime that match their preferences. 
# Use the following pieces of context to answer the question at the end. 
# For each question, suggest three anime, with a short description of the plot and the reason why the user migth like it.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Your response:"""


# PROMPT = PromptTemplate(
#     template=template, input_variables=["context", "question"])

# chain_type_kwargs = {"prompt": PROMPT}

# llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0) 

# qa = RetrievalQA.from_chain_type(llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)

# query = "I'm looking for an action anime with animals, any suggestions?"
# result = qa({'query':query})
# print(result['result'])