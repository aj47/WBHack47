from sentence_transformers import SentenceTransformer
from aperturedb import Connector
import numpy as np

def connect_to_db():
    db = Connector.Connector(host="metis-t3oknmxh.farm0000.cloud.aperturedata.io",
                            user="admin",
                            password="metisadmin123!")
    return db

def find_closest_descriptors(db, descriptorset_name, query_embedding, k=20):
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

def search_pdf(query_text, descriptorset_name):
    # Connect to DB
    db = connect_to_db()
    
    # Initialize model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Convert query to embedding
    query_embedding = np.array(model.encode(query_text))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Find closest matches
    matches = find_closest_descriptors(db, descriptorset_name, query_embedding)
    
    # Process and return results
    results = []
    for match in matches:
        results.append({
            'pdf_name': match['pdf_name'],
            'text': match['text'],
            'similarity': match['_distance']  # Convert distance to similarity
        })
    
    # Return the result with highest similarity score
    if results:
        return max(results, key=lambda x: x['similarity'])
    return None

if __name__ == "__main__":
    # Example usage
    query = "Create New User"
    descriptorset_name = "pdf_instructions_correct2"
    result = search_pdf(query, descriptorset_name)
    

    print("\nTop Search Result:")
    if result:
        print(f"\nPDF Name: {result['pdf_name']}")
        print(f"Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.2f}")
    else:
        print("No results found.")