from typing import Any
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv
import agent
import asyncio
import os

load_dotenv()  # reads .env into environment
api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
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
llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",           # or any model tag supported by your OpenRouter account
    temperature=0.2,
    openai_api_base="https://openrouter.ai/api/v1",          # picks up from OPENAI_API_BASE
    openai_api_key=api_key , 
    streaming=True,
    callback_manager=manager
               
)


app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory="templates")

# Optional: Mount static files if you have any (CSS, JS, images)
# app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/chat")
async def text_chatbot(message: ChatMessage):
    try:
        
        agent_instance = agent.create_agent_with_tools(llm=llm, pdf_path="pdfs/dubai_tourist_guide.pdf")
        if agent_instance is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create agent instance"}
            )
        
            
        task = asyncio.create_task(agent_instance.arun({"question": message.message}))
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



