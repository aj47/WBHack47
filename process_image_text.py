import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import fitz  # PyMuPDF
import io
import numpy as np

import PyPDF2
from aperturedb import Connector as Connector
import re
import os

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
            "dimensions": 1024,
            "engine": "Flat",
            "metric": "IP",
        }
    }]
    db.query(q)

def initialize_siglip_model():
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-384')
    tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-384')
    return model, preprocess, tokenizer

def encode_image(image, model, preprocess):
    """Encode an image using SigLIP model"""
    image_input = preprocess(image).unsqueeze(0)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_input)
        image_features = F.normalize(image_features, dim=-1)
    
    return image_features.cpu().numpy()[0]

def encode_text(text, model, tokenizer):
    """Encode text using SigLIP model"""
    text_input = tokenizer([text], context_length=model.context_length)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_input)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features.cpu().numpy()[0]

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF"""
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return images

def add_text_descriptor(db, text_embedding, text, descriptorset_name, pdf_name):
    embedding_bytes = text_embedding.astype('float32').tobytes()
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

def add_image_descriptor(db, image_embedding, image_index, descriptorset_name, pdf_name):
    embedding_bytes = image_embedding.astype('float32').tobytes()
    q = [{
        "AddDescriptor": {
            "set": descriptorset_name,
            "properties": {
                "pdf_name": pdf_name,
                "image_index": image_index,
            },
        }
    }]

    responses, blobs = db.query(q, [embedding_bytes])

def process_pdf(pdf_path, descriptorset_name, db, model, preprocess, tokenizer):    
    

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Process text
    texts = extract_headings_from_pdf(pdf_path)

    # Process images
    images = extract_images_from_pdf(pdf_path)

    if pdf_name == "invite_team_member": images.remove(images[1])
    else: images.remove(images[0])

    image_features = []
    text_features = []

    output_dir = "extracted_images"
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{pdf_name}_image_{i}.png"))
        image_features.append(encode_image(image, model, preprocess))

    for text in texts:
        text_features.append(encode_text(text, model, tokenizer))

    for text_feature, txt in zip(text_features, texts):
        add_text_descriptor(db, text_feature, txt, descriptorset_name, pdf_name)
    
    for image_feature, image_index in zip(image_features, range(len(images))):
        add_image_descriptor(db, image_feature, image_index, descriptorset_name, pdf_name)
    
    print(f"Processed {len(text)} headings and {len(images)} images from PDF {pdf_name}")

if __name__ == "__main__":
    # Example usage

    # Get current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PDF file names
    pdf_files = [
        "create_organization.pdf",
        "create_new_user.pdf",
        "invite_team_member.pdf",
        "create_new_project.pdf"
    ]
    
    # Create full paths
    pdf_paths = [os.path.join(current_dir, pdf_file) for pdf_file in pdf_files]

    # Connect to DB
    db = connect_to_db()

    # Initialize models
    model, preprocess, tokenizer = initialize_siglip_model()

    # Ensure descriptor set exists
    descriptorset_name = "pdf_instructions_image_text"
    add_descriptor_set(db, descriptorset_name)

    for pdf_path in pdf_paths:
        process_pdf(pdf_path, descriptorset_name, db, model, preprocess, tokenizer)