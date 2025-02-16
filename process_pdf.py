import PyPDF2
from sentence_transformers import SentenceTransformer
from aperturedb import Connector as Connector
import re
import os
import numpy as np

def connect_to_db():
    db = Connector.Connector(host="metis-t3oknmxh.farm0000.cloud.aperturedata.io",
                            user="admin",
                            password="metisadmin123!")
    return db

def clean_heading(text):
    """Clean up a heading by removing unwanted text and whitespace"""
    # Remove the "Made with Scribe" text
    text = re.sub(r'Made with Scribe.*?com\n*', '', text)
    # Remove numbered prefixes
    text = re.sub(r'^\d+\.?\s*', '', text)
    # Remove extra whitespace, newlines, and tabs
    text = ' '.join(text.split())
    return text

def extract_headings_from_pdf(pdf_path):
    """Extract text sections from PDF"""
    headings = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            # Split text into sections
            sections = re.split(r'\n(?=[A-Z][^a-z]*\n|[0-9]+\.)', text)
            # Clean each section and filter out empty ones
            cleaned_sections = [clean_heading(section) for section in sections]
            headings.extend([section for section in cleaned_sections 
                           if section.strip() and 
                           section != "https://scribehow.com" and
                           len(section) > 5])  # Filter out very short sections
    return headings

def add_descriptor_set(db, descriptorset_name):
    q = [{
        "AddDescriptorSet": {
            "name": descriptorset_name,
            "dimensions": 384,
            "engine": "Flat",
            "metric": "IP",
        }
    }]
    db.query(q)

def add_descriptor(db, text, model, descriptorset_name, pdf_name):
    embedding = np.array(model.encode(text))
    embedding = embedding / np.linalg.norm(embedding)
    embedding_bytes = embedding.astype('float32').tobytes()
    q = [{
        "AddDescriptor": {
            "set": descriptorset_name,
            "properties": {
                "pdf_name": pdf_name,
                "text": text,
            },
        }
    }]

    responses, blobs = db.query(q, [embedding_bytes])

def process_pdf(pdf_path, descriptorset_name, db):    
    # Initialize model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract and process headings
    headings = extract_headings_from_pdf(pdf_path)
    
    # Add each heading as a descriptor
    for heading in headings:
        add_descriptor(db, heading, model, descriptorset_name, pdf_name)
    
    print(f"Processed {len(headings)} headings from PDF {pdf_name}")

if __name__ == "__main__":
    # Example usage
    pdf_paths = ["/Users/yuvanshuagarwal/Desktop/Tech_SF/Luma_AI_Hackathon/WBHack47/create_organization.pdf",
                "/Users/yuvanshuagarwal/Desktop/Tech_SF/Luma_AI_Hackathon/WBHack47/create_new_user.pdf"]

    # Connect to DB
    db = connect_to_db()

    # Ensure descriptor set exists
    descriptorset_name = "pdf_instructions_correct2"
    add_descriptor_set(db, descriptorset_name)

    for i in range(len(pdf_paths)):
        process_pdf(pdf_paths[i], descriptorset_name, db)