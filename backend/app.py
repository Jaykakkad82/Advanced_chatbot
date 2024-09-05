from flask import Flask, request, jsonify
import openai
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.vectorstores import Neo4jVectorStore
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

app = Flask(__name__)

# Initialize Neo4j database for storing the knowledge graph
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
graph_db = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# session
conversations = {}


# Function to extract text and tables from PDFs
def extract_text_and_tables(pdf_file):
    text = ""
    tables = []
    
    # Step 1: Extract Text Using PyMuPDF (fitz)
    try:
        with fitz.open(pdf_file) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text")  # Extract plain text from the page
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
    
    # Step 2: Extract Tables Using pdfplumber
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted_tables = page.extract_tables()  # Extract tables as a list of rows
                for table in extracted_tables:
                    # Convert the table into a Pandas DataFrame for structured handling
                    df = pd.DataFrame(table[1:], columns=table[0])  # Assuming the first row is the header
                    tables.append(df)
    except Exception as e:
        print(f"Error extracting tables from {pdf_file}: {e}")
    
    return text, tables

def store_in_graph_db(graph_data, tables_data):
    with graph_db.session() as session:
        # Storing text-based knowledge graph
        for node, edges in graph_data.items():
            session.run("MERGE (n:Node {name: $node})", node=node)
            for edge, target in edges.items():
                session.run("MATCH (n1:Node {name: $node}), (n2:Node {name: $target}) "
                            "MERGE (n1)-[:REL {type: $edge}]->(n2)",
                            node=node, target=target, edge=edge)
        
        # Storing tables as structured nodes in the graph
        for table_index, table_df in enumerate(tables_data):
            table_name = f"Table_{table_index + 1}"
            # Create a node for the table
            session.run("MERGE (t:Table {name: $table_name})", table_name=table_name)
            
            for row_index, row in table_df.iterrows():
                row_node_name = f"{table_name}_Row_{row_index + 1}"
                session.run("MERGE (r:Row {name: $row_node_name})", row_node_name=row_node_name)
                
                # Link each row to the table
                session.run("MATCH (t:Table {name: $table_name}), (r:Row {name: $row_node_name}) "
                            "MERGE (t)-[:HAS_ROW]->(r)", table_name=table_name, row_node_name=row_node_name)
                
                # For each row, create nodes for columns and link them
                for col_name, col_value in row.items():
                    if pd.isna(col_value):
                        col_value = "N/A"
                    column_node_name = f"{row_node_name}_{col_name}"
                    session.run("MERGE (c:Column {name: $column_node_name, value: $col_value})",
                                column_node_name=column_node_name, col_value=str(col_value))
                    session.run("MATCH (r:Row {name: $row_node_name}), (c:Column {name: $column_node_name}) "
                                "MERGE (r)-[:HAS_COLUMN]->(c)", row_node_name=row_node_name, column_node_name=column_node_name)

@app.route('/api/v1/upload-pdfs', methods=['POST'])
def upload_pdfs():
    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files uploaded"}), 400

    session_id = request.form.get('session_id', 'default')  # Use 'default' if no session ID provided
    
    # Check if the session exists and if an API key is stored for the session
    if session_id not in conversations or 'api_key' not in conversations[session_id]:
        return jsonify({"error": "No API key found for this session"}), 400
    
    api_key = conversations[session_id]['api_key']
    
    # Extracting text and tables logi
    pdf_files = request.files.getlist('pdfs')
    
    # Extract text and tables from PDFs
    all_text = ""
    all_tables = []
    for pdf_file in pdf_files:
        text, tables = extract_text_and_tables(pdf_file)
        all_text += text
        all_tables.extend(tables)

    # Create a knowledge graph using LLMGraphTransformer for the text data
    graph_transformer = LLMGraphTransformer()
    knowledge_graph = graph_transformer.build_graph(all_text)

    # Store both text and tables in Neo4j
    store_in_graph_db(knowledge_graph, all_tables)

    return jsonify({"message": "PDFs uploaded and processed successfully", "session_id": session_id}), 200

    # return jsonify({"message": "PDFs processed and knowledge graph created"}), 200

