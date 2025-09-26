# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from openai import OpenAI
# from typing import Optional

# app = FastAPI(title="OpenAI Chat API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     developer_message: str
#     user_message: str
#     model: Optional[str] = "gpt-4.1-mini"
#     api_key: str

# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     try:
#         client = OpenAI(api_key=request.api_key)

#         def generate():
#             try:
#                 stream = client.chat.completions.create(
#                     model=request.model,
#                     messages=[
#                         {"role": "system", "content": request.developer_message},
#                         {"role": "user", "content": request.user_message}
#                     ],
#                     stream=True
#                 )

#                 for chunk in stream:
#                     content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
#                     if content:
#                         yield content

#                 print(content)

#             except Exception as e:
#                 yield f"\n[ERROR] Streaming interrupted: {str(e)}"

#         return StreamingResponse(generate(), media_type="text/event-stream")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/health")
# async def health_check():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn


#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()

# Allow CORS from same domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Since we're on the same domain, we can be more permissive
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-4"
    api_key: str

@app.post("/api/chat")
async def chat(request: PromptRequest):
    try:
        client = OpenAI(api_key=request.api_key)

        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "user", "content": request.prompt}
            ]
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}