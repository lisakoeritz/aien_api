#from modal import Image, App, Secret
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.rag import get_answer_and_docs

app = FastAPI(title="AI Ethics Navigator API", description="API for the AI Ethics Navigator", version="0.1")

#Add CORS middleware
origins = [
    "https://ai-ethics-navigator.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/api/qa", description="Ask a question to the AI Ethics Navigator")
def qa(q: Question):
    response = get_answer_and_docs(question=q.question)
    response_dict = {
        "question": q.question,
        "answer": response["answer"],
        "documents": [doc.dict() for doc in response["context"]]
    }
    return JSONResponse(content=response_dict, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)