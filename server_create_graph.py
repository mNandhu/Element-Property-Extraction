# Import Libraries

import pickle
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama

from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split Document into Chunks
from langchain_community.document_loaders import DirectoryLoader  # Load md files

markdown_path = "data/markdowns"
loader = DirectoryLoader(markdown_path, glob="*.md")
documents = loader.load()
print(f"Loaded {len(documents)} documents from {markdown_path}")

chunk_size = 750
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                               length_function=len, add_start_index=True, keep_separator=False,
                                               strip_whitespace=True)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

document = chunks[10]
print(document.page_content)
print(document.metadata)

## Create Graph Documents
llm = ChatOllama("llama3.1:70b")
llm_transformer = LLMGraphTransformer(llm=llm, allowed_nodes=["ALLOY", "ELEMENT", "PROPERTY_NAME", "PROPERTY_VALUE"])

graph_documents = llm_transformer.convert_to_graph_documents(chunks)
print(graph_documents[0])

# Save Graph Documents
with open("Nodes.bin", "wb") as f:  # Save graph_docs obj to file
    pickle.dump(graph_documents, f)
    

# Graph to Neo4j
#
# from neo4j import GraphDatabase
# from langchain_community.graphs import Neo4jGraph
#
# with open("graph_docs.bin", "rb") as f:  # Load graph_docs from file
#     retrieved_graph_documents = pickle.load(f)
#
# graph = Neo4jGraph()
# graph.add_graph_documents(
#     retrieved_graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )