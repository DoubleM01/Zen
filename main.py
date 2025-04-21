from typing import Any, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
# langchain huggingface
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv
import uvicorn
import agent
import asyncio
import os
import location_search
import subprocess
from huggingface_hub import InferenceClient
import pandas as pd
import pickle
import time
import json

load_dotenv()  # reads .env into environment
api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
load_dotenv()

class SSEStreamHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs):
        # enqueue each new token
        self.queue.put_nowait(token)
        #print(token)

    def on_llm_end(self, response: Any = None, **kwargs):
        # signal end of stream
        self.queue.put_nowait("[DONE]")


queue: asyncio.Queue[str] = asyncio.Queue()

# wire up the handler
handler = SSEStreamHandler(queue)
manager = CallbackManager([handler])

# Initialize OpenAI client
llm = ChatOpenAI(
    model="deepseek/deepseek-v3-base:free",           # or any model tag supported by your OpenRouter account
    # deepseek/deepseek-chat:free
    #deepseek-ai/DeepSeek-R1 huggingface
    temperature=0.3,
    #
    openai_api_base="https://openrouter.ai/api/v1",          # picks up from OPENAI_API_BASE
    openai_api_key=api_key , 
    streaming=True,
    callback_manager=manager
               
)


app = FastAPI()
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Store the user's last known location
user_location: Dict[str, Optional[float]] = {"lat": None, "lon": None}

class Location(BaseModel):
    lat: float
    lon: float

@app.post("/report_location")
async def report_location(loc: Location):
    # Here you can store loc in DB or compute nearest gym, route, etc.
    try:
        print(f"Received location: {loc.lat}, {loc.lon}")
        # Store the user's location
        user_location["lat"] = loc.lat
        user_location["lon"] = loc.lon
        
        radius: int = 3000  # Search radius in meters
        gyms = location_search.get_nearby_gyms(loc.lat, loc.lon, radius)
        
        # Format the gym data for the frontend
        formatted_gyms = []
        for gym in gyms:
            if len(gym) >= 5:  # Make sure we have all the data (name, lat, lon, distance, walking_time)
                name, lat, lon, distance, walking_time = gym
                formatted_gyms.append({
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "distance": distance,  
                    "walking_time": walking_time
                })
            else:
                # Fallback for old format
                name, lat, lon = gym[:3]
                formatted_gyms.append({
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "distance": None,
                    "walking_time": None
                })
                
        return {"received": loc, "gyms": formatted_gyms}
    except Exception as e:
        print(f"Error processing location: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Mount templates
templates = Jinja2Templates(directory="templates")


class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/chat")
async def text_chatbot(message: ChatMessage):
    try:
        
        agent_instance = agent.create_agent_with_tools(llm=llm, pdf_path=None)
        if agent_instance is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create agent instance"}
            )
        
        # Check if the message is asking about finding a gym
        user_message = message.message.lower()
        if any(keyword in user_message for keyword in ["gym", "fitness", "workout", "exercise"]) and any(keyword in user_message for keyword in ["near", "nearby", "closest", "find", "where", "location"]):
            # If the user has shared their location before, we can use it
            if user_location["lat"] is not None and user_location["lon"] is not None:
                # Use the stored location to find gyms
                lat = user_location["lat"]
                lon = user_location["lon"]
                gyms = location_search.get_nearby_gyms(lat, lon)
                
                if gyms and len(gyms) > 0:
                    # Format gym list with distance and walking time
                    gym_list = []
                    for i, gym in enumerate(gyms):
                        if len(gym) >= 5:  # New format with distance and walking time
                            name, gym_lat, gym_lon, distance, walking_time = gym
                            gym_list.append(f"{i+1}. {name} - {distance:.1f}m away ({walking_time} min walk)")
                        else:  # Old format without distance and walking time
                            name, gym_lat, gym_lon = gym[:3]
                            gym_list.append(f"{i+1}. {name} (Lat: {gym_lat}, Lon: {gym_lon})")
                    
                    gym_list_text = "\n".join(gym_list)
                    response = f"I found these gyms near your location (Lat: {lat}, Lon: {lon}):\n\n{gym_list_text}\n\nYou can click the 'Find Nearest Gym' button to see these locations on the map and select the closest one. The closest gym is {gyms[0][0]}, which is {gyms[0][3]:.1f}m away and would take approximately {gyms[0][4]} minutes to walk to. return only the answer direclty, show the explanation separately"
                else:
                    response = "I couldn't find any gyms near your location. You might want to try increasing the search radius or check a different area. return only the answer direclty, show the explanation separately"
                
                task = asyncio.create_task(agent_instance.arun(response))
            else:
                # Ask the user to share their location
                task = asyncio.create_task(agent_instance.arun(
                    "I'd be happy to help you find a nearby gym! To do that, I'll need your location. Please click the 'Share Location for Local Wellness Resources' button below or the 'Find Nearest Gym' button to share your location. Alternatively, you can tell me your latitude and longitude coordinates. return only the answer direclty, show the explanation separately"
                ))
        else:
            # For other queries, use the agent normally
            msg = message.message 
            task = asyncio.create_task(agent_instance.arun(msg))
            
        async def event_generator():
            while True:
                token = await queue.get()
                print(token)
                # yield in SSE format
                yield f"data: {token}\n\n"
                if token == "[DONE]":
                    break
            # ensure the background task finishes
            await task
        
    # 6. Return a StreamingResponse with 'text/event-stream'
        return StreamingResponse(event_generator(), media_type="text/event-stream")
      
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )



# MET values for activities (copied from act_model.py)
MET_VALUES = {
    'STANDING': 1.3,
    'SITTING': 1.0,
    'LAYING': 1.0,
    'WALKING': 3.3,
    'WALKING_DOWNSTAIRS': 3.5,
    'WALKING_UPSTAIRS': 4.0
}

def load_activity_model():
    """Load the activity recognition model from best_model.pkl"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_csv_data(csv_path='test.csv'):
    """Load all rows from the CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Drop non-feature columns if they exist
        X_test = df.drop(columns=['subject', 'Activity', 'ActivityName'], errors='ignore')
        return X_test
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None

def calories_from_met(activity: str, weight_kg: float, duration_sec: float) -> float:
    """
    Estimate calories burned based on MET.
    
    - activity: one of the keys in MET_VALUES
    - weight_kg: user weight in kilograms
    - duration_sec: duration of the activity window in seconds
    
    Formula: Calories = MET × 3.5 × weight_kg / 200  (kcal per minute)
             then × (duration_sec / 60) to scale to the window length.
    """
    met = MET_VALUES.get(activity.upper(), 1.0)
    kcal_per_min = met * 3.5 * weight_kg / 200
    return kcal_per_min * (duration_sec / 60)

@app.get("/stream_predictions")
async def stream_predictions():
    """Stream predictions from all rows in test.csv, one row per second, repeating forever"""
    
    # Load the model
    model = load_activity_model()
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load activity model"}
        )
    
    # Load the data
    data = load_csv_data()
    if data is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load CSV data"}
        )
    
    # Default weight for calorie calculation
    weight_kg = 70  # Default weight in kg
    window_duration = 1  # 1 second per prediction
    
    async def generate_predictions():
        """Generate predictions one row at a time, with a 1-second delay between rows, repeating forever"""
        while True:  # Infinite loop to keep streaming data
            for index, row in data.iterrows():
                # Make prediction
                row_df = pd.DataFrame([row])
                prediction = model.predict(row_df)[0]
                
                # Calculate calories
                calories = calories_from_met(prediction, weight_kg, window_duration)
                
                # Create result object
                result = {
                    "row": int(index),
                    "prediction": prediction,
                    "calories": round(calories, 4),
                    "features": row.to_dict()
                }
                
                # Send the result as a server-sent event
                yield f"data: {json.dumps(result)}\n\n"
                
                # Wait for 1 second before sending the next row
                await asyncio.sleep(1)
            
            # No 'complete' status is sent, as we want to continue indefinitely
    
    return StreamingResponse(generate_predictions(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem"
    )





