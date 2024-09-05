from flask import Flask, request, jsonify
import openai
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.vectorstores import Neo4jVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from networkx import DiGraph
from neo4j import GraphDatabase
import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd