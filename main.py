from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ingest import VectorIngestor
from app import AIVoiceAssistant
import speech_recognition as sr
from fastapi.responses import JSONResponse
from uuid import uuid4

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images)
app.mount("/images", StaticFiles(directory="images"), name="images")

# Initialize vector ingestor
image_folder = "./images"
ingestor = VectorIngestor(image_folder)

# Initialize voice assistant
vector_db_url = "http://localhost:6333"
ai71_api_key = os.getenv("AI71_API_KEY")
assistant = AIVoiceAssistant(vector_db_url, ai71_api_key)

class QueryRequest(BaseModel):
    query_text: str

# Define CartItem model
class CartItem(BaseModel):
    id: str
    name: str
    description: str
    price: float
    quantity: int = 1

# In-memory cart storage (for demonstration; use a database for production)
cart = []

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Voice Assistant API"}

@app.post("/ingest/")
async def ingest_images(files: List[UploadFile] = File(...)):
    image_metadata = {
        "th.jpg": {
            "description": "A high-quality pink summer t-shirt, designed with a long cut and crafted from premium fabric for maximum comfort and durability. Price is $100",
            "price": 100.0
        },
        "thi.jpg": {
            "description": "A vibrant black summer t-shirt, featuring a long design and made from breathable fabric, ideal for casual outings and hot weather. Price is $100",
            "price": 100.0
        },
        "pant1.jpg": {
            "description": "A well-crafted pair of quality pants, offering a perfect fit and made from durable fabric for everyday wear and comfort. Price is $120",
            "price": 120.0
        },
        "pant.jpg": {
            "description": "Comfortable summer pants, made from elastic fabric, providing a relaxed fit and easy movement for warm weather activities. Price is $121",
            "price": 121.0
        },
        "womentshirt.jpg": {
            "description": "A stylish women t-shirt with a long cut, made from soft fabric and designed for both comfort and elegance, suitable for various occasions. Price is $140",
            "price": 140.0
        },
        "womentshirt1.jpg": {
            "description": "A blue summer t-shirt for women, featuring a long design and crafted from high-quality fabric, perfect for staying cool and fashionable. Price is $150",
            "price": 150.0
        },
        "trouser.jpg": {
            "description": "A comfortable black summer trouser, designed with a long cut and made from soft fabric, ideal for casual wear and everyday use. Price is $78",
            "price": 78.0
        },
        "trouser1.jpg": {
            "description": "A versatile blue summer trouser with a long fit, made from high-quality fabric for a stylish and comfortable look. Price is $72",
            "price": 72.0
        }
    }
    
    images = []
    for file in files:
        file_path = os.path.join(image_folder, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Generate a unique ID for the image
        image_id = str(uuid4())
        metadata = image_metadata.get(file.filename, {"description": "No description available.", "price": 0.0})
        description = metadata["description"]
        price = metadata["price"]
        
        # Save image metadata
        images.append({
            "id": image_id,
            "file_name": file.filename,
            "description": description,
            "price": price
        })
    
    ingestor.create_vector_db(images)
    return {"message": "Images ingested successfully", "images": images}

@app.post("/query/")
async def query_images(request: QueryRequest):
    # Fetch the search results from the vector database
    search_results = assistant.query_vector_db(request.query_text)

    # Construct the filtered images list from search results
    filtered_images = []
    for result in search_results:
        payload = result.payload
        image_id = payload.get("id", "unknown_id")
        file_name = payload.get("file_name", "unknown.jpg")
        description = payload.get("description", "No description")
        price = payload.get("price", 0.0)

        filtered_images.append({
            "id": image_id,
            "filename": file_name,
            "description": description,
            "price": price
        })

    # Determine the item to add to cart (for example, the first match)
    add_to_cart_item = filtered_images[0] if filtered_images else None

    # Check if the user query contains a command to add or delete an item
    if "add to cart" in request.query_text.lower() and add_to_cart_item:
        cart_item = CartItem(
            id=add_to_cart_item["id"],
            name=add_to_cart_item["description"],
            description=add_to_cart_item["description"],
            price=add_to_cart_item["price"]
        )
        cart.append(cart_item)
        action_result = "added_to_cart"
    elif "delete from cart" in request.query_text.lower() and add_to_cart_item:
        delete_from_cart_by_id(add_to_cart_item["id"])
        action_result = "deleted_from_cart"
    else:
        action_result = "no_item_found"

    # Generate response with action result
    response = assistant.generate_response(request.query_text, action_result)

    return {
        "response": response,
        "images": filtered_images,
        "addToCart": add_to_cart_item if action_result == "added_to_cart" else None
    }

def delete_from_cart_by_id(item_id: str):
    global cart
    cart = [item for item in cart if item.id != item_id]

@app.post("/voice-query/")
async def voice_query(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio_file_path = os.path.join("audio", file.filename)

    with open(audio_file_path, "wb") as buffer:
        buffer.write(file.file.read())

    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio)
            response = assistant.handle_user_query(query_text)
            return {"response": response, "query_text": query_text}
        except sr.UnknownValueError:
            return JSONResponse(status_code=400, content={"message": "Could not understand the audio."})
        except sr.RequestError as e:
            return JSONResponse(status_code=500, content={"message": f"Could not request results from the speech recognition service; {e}"})

@app.post("/cart/add/")
async def add_to_cart(item: CartItem):
    cart.append(item)
    return {"message": "Item added to cart", "cart": cart}

@app.post("/cart/delete/")
async def delete_from_cart(item_id: str):
    global cart
    cart = [item for item in cart if item.id != item_id]
    return {"message": "Item removed from cart", "cart": cart}

@app.post("/cart/edit/")
async def edit_cart_item(item: CartItem):
    global cart
    for i, cart_item in enumerate(cart):
        if cart_item.id == item.id:
            cart[i] = item
            return {"message": "Item updated in cart", "cart": cart}
    return {"message": "Item not found in cart"}

@app.get("/cart/")
async def get_cart():
    return {"cart": cart}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
