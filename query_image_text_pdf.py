import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from aperturedb import Connector
import numpy as np

def connect_to_db():
    db = Connector.Connector(host="metis-t3oknmxh.farm0000.cloud.aperturedata.io",
                            user="admin",
                            password="metisadmin123!")
    return db

def initialize_siglip_model():
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-384')
    tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-384')
    return model, preprocess, tokenizer

def encode_text_query(text, model, tokenizer):
    """Encode text using SigLIP model"""
    text_input = tokenizer([text], context_length=model.context_length)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_input)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features.cpu().numpy()[0]

def find_closest_descriptors(db, descriptorset_name, query_embedding, k=3):
    """Find k closest matches to the query"""
    query_embedding_bytes = query_embedding.astype('float32').tobytes()
    q = [{
        "FindDescriptor": {
            "set": descriptorset_name,
            "k_neighbors": k,
            "distances": True,
            "labels": True,
            "results": {
                "all_properties": True
            }
        }
    }]
    
    responses, blobs = db.query(q, [query_embedding_bytes])
    return responses[0]["FindDescriptor"]["entities"]

def search_pdf(query_text):
    # Connect to DB
    db = connect_to_db()

    
    descriptorset_name = "pdf_instructions_image_text"
    
    # Initialize SigLIP model
    model, _, tokenizer = initialize_siglip_model()
    
    # Convert query to embedding using SigLIP
    query_embedding = encode_text_query(query_text, model, tokenizer)
    
    # Find closest matches
    matches = find_closest_descriptors(db, descriptorset_name, query_embedding)
    
    # Process and return results
    results = []
    for match in matches:
        result_dict = {
            'pdf_name': match['pdf_name'],
            'similarity': match['_distance']
        }
        # Add text or image_info depending on what's available
        if 'text' in match:
            result_dict['text'] = match['text']
            result_dict['type'] = 'text'
        elif 'image_index' in match:
            result_dict['image_index'] = match['image_index']
            result_dict['type'] = 'image'
        results.append(result_dict)
    
    # Return the result with highest similarity score
    if results:
        return max(results, key=lambda x: x['similarity'])
    return None

if __name__ == "__main__":
    # Example usage
    query = "Create New User"
    result = search_pdf(query)

    print("\nTop Search Result:")
    if result:
        print(f"\nPDF Name: {result['pdf_name']}")
        if result['type'] == 'text':
            print(f"Text: {result['text']}")
        else:
            print(f"Image Index: {result['image_index']}")
        print(f"Type: {result['type']}")
        print(f"Similarity: {result['similarity']:.2f}")
    else:
        print("No results found.")
    
    