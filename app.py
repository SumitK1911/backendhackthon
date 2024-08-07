import os
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from PIL import Image
from ai71 import AI71
import speech_recognition as sr
import pyttsx3

class AIVoiceAssistant:
    def __init__(self, vector_db_url, ai71_api_key):
        self.vector_db_client = QdrantClient(url=vector_db_url, prefer_grpc=False)
        self.collection_name = "image_vectors"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ai71_client = AI71(api_key=ai71_api_key)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            features = outputs.detach().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def query_vector_db(self, query_text, top_k=3):
        # Convert query text to image features using CLIP model
        try:
            inputs = self.processor(text=query_text, return_tensors="pt")
            text_features = self.model.get_text_features(input_ids=inputs['input_ids'])
            text_features = text_features.detach().numpy().flatten()
            
            # Query the vector database
            search_result = self.vector_db_client.search(
                collection_name=self.collection_name,
                query_vector=text_features,
                limit=top_k
            )
            return search_result
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return []

    def generate_response(self, query_text, action_result=None):
        search_results = self.query_vector_db(query_text)
        
        if not search_results:
            return "I couldn't find any relevant information."

        # Combine metadata from search results
        combined_metadata = " ".join([result.payload.get("description", "") for result in search_results])
        if action_result == "added_to_cart":
            prompt = f"Based on the information provided, {combined_metadata}, the item has been successfully added to your cart."
        elif action_result == "no_item_found":
            prompt = f"Based on the information provided, {combined_metadata}, no relevant item was found to add to your cart."
        elif "delete" in query_text.lower() or "remove" in query_text.lower():
           item_description = query_text.lower().replace("delete the", "").replace("remove the", "").strip()
           prompt = f"Delete the item with description '{item_description}' from the cart."
        else:
           prompt = f"Based on the information provided, {combined_metadata}, answer the user query: {query_text}"
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant who provides information based on the provided context."},
                {"role": "user", "content": prompt}
            ]
            
            content = ""
            for chunk in self.ai71_client.chat.completions.create(
                messages=messages,
                model="tiiuae/falcon-180B-chat",
                stream=True,
            ):
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    content += delta_content
                  
            
            return content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response."

    def handle_user_query(self, query_text):
        return self.generate_response(query_text)
    
    def speak_text(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_to_user(self):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
                query_text = self.recognizer.recognize_google(audio)
                print(f"User: {query_text}")
                return query_text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

if __name__ == "__main__":
    vector_db_url = "http://localhost:6333"
    ai71_api_key = os.getenv("AI71_API_KEY")  # Ensure this is set in your environment variables
    
    assistant = AIVoiceAssistant(vector_db_url, ai71_api_key)
    
    while True:
        user_query = assistant.listen_to_user()
        if "terminate" in user_query.lower():
            print("Terminating...")
            assistant.speak_text("Goodbye!")
            break
        
        response = assistant.handle_user_query(user_query)
        print(f"AI Response: {response}")
        assistant.speak_text(response)
