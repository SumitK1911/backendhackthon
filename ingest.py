import os
import uuid
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

class VectorIngestor:
    def __init__(self, image_folder, url="http://localhost:6333"):
        self.image_folder = image_folder
        self.url = url
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.client = QdrantClient(url=self.url, prefer_grpc=False)
        self.collection_name = "image_vectors"

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            features = outputs.detach().numpy().flatten()  # Flatten the features
            print(f"Extracted features for {image_path} with shape: {features.shape}")
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def create_vector_db(self, images):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
        
        try:
            vector_params = VectorParams(size=512, distance="Cosine")
            self.client.create_collection(collection_name=self.collection_name, vectors_config=vector_params)
            print(f"Created new collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
        
        points = []
        for image in images:
            image_id = image['id']
            image_name = image['file_name']
            description = image['description']
            price = image['price']
            
            print(f"Processing image: {image_name} with description: {description} and price: {price}")
            image_path = os.path.join(self.image_folder, image_name)
            if os.path.isfile(image_path):
                features = self.extract_features(image_path)
                if features is not None and features.shape[0] == 512:
                    points.append({
                        "id": image_id,
                        "vector": features.tolist(),
                        "payload": {
                            "file_name": image_name,
                            "description": description,
                            "price": price
                        }
                    })
                    print(f"Added {image_name} to points.")
                else:
                    print(f"Skipping {image_name}: expected 512 dimensions but got {features.shape[0] if features is not None else 'None'}")
            else:
                print(f"File {image_path} does not exist or is not a file.")

        try:
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"Inserted {len(points)} points into the vector database.")
            else:
                print("No valid points to upsert.")
        except Exception as e:
            print(f"Error upserting points: {e}")

if __name__ == "__main__":
    image_folder = "./images"
    image_metadata = {
        "th.jpg": {
            "description": "A high-quality pink summer t-shirt, designed with a long cut and crafted from premium fabric for maximum comfort and durability.price is $100",
            "price": 100.0
        },
        "thi.jpg": {
            "description": "A vibrant black summer t-shirt, featuring a long design and made from breathable fabric, ideal for casual outings and hot weather.price is $100",
            "price": 100.0
        },
        "pant1.jpg": {
            "description": "A well-crafted pair of quality pants, offering a perfect fit and made from durable fabric for everyday wear and comfort.price is $120",
            "price": 120.0
        },
        "pant.jpg": {
            "description": "Comfortable summer pants, made from elastic fabric, providing a relaxed fit and easy movement for warm weather activities. price is $121",
            "price": 121.0
        },
        "womentshirt.jpg": {
            "description": "A stylish women t-shirt with a long cut, made from soft fabric and designed for both comfort and elegance, suitable for various occasions.price is $140",
            "price": 140.0
        },
        "womentshirt1.jpg": {
            "description": "A blue summer t-shirt for women, featuring a long design and crafted from high-quality fabric, perfect for staying cool and fashionable.price is $150",
            "price": 150.0
        },
        "trouser.jpg": {
            "description": "A comfortable black summer trouser, designed with a long cut and made from soft fabric, ideal for casual wear and everyday use. price is $78",
            "price": 78.0
        },
        "trouser1.jpg": {
            "description": "A versatile blue summer trouser with a long fit, made from high-quality fabric for a stylish and comfortable look. price is $72",
            "price": 72.0
        }
    }

    images = [{
        "id": str(uuid.uuid4()),
        "file_name": file_name,
        "description": metadata["description"],
        "price": metadata["price"]
    } for file_name, metadata in image_metadata.items()]

    ingestor = VectorIngestor(image_folder)
    ingestor.create_vector_db(images)
