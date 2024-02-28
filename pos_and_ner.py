import streamlit as st
import requests
import spacy
from bs4 import BeautifulSoup

# Load pre-trained models for POS and NER
nlp_pos = spacy.load("en_core_web_sm")
nlp_ner = spacy.load("en_core_web_sm")

# Function to extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

# Function for POS tagging
def pos_tagging(text):
    doc = nlp_pos(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

# Function for Named Entity Recognition (NER)
def named_entity_recognition(text):
    doc = nlp_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI
st.title("POS and NER Tool")

# Sidebar
st.sidebar.title("Select Data Input")
data_input = st.sidebar.radio("Select Data Source:", ("Text", "URL"))

# Main content
st.header("Input Text")

input_text = ""  # Initialize input_text variable

if data_input == "Text":
    input_text = st.text_area("Paste your text here:")
else:
    url = st.text_input("Enter URL:")
    if st.button("Extract Text"):
        input_text = extract_text_from_url(url)
        st.text_area("Extracted Text:", input_text)

# Select task
task = st.sidebar.radio("Select Task:", ("POS Tagging", "Named Entity Recognition"))

# Perform task based on selection
if st.button("Perform Task"):
    if data_input == "URL":
        input_text = extract_text_from_url(url)
    
    if task == "POS Tagging":
        pos_tags = pos_tagging(input_text)
        st.header("POS Tagging Results")
        st.table(pos_tags)
    else:
        entities = named_entity_recognition(input_text)
        st.header("Named Entity Recognition Results")
        st.table(entities)


