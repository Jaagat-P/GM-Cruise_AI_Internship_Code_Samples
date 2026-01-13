from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import uuid
from typing import Dict, Any
import json

from models.whisper_model import WhisperTranscriber
from models.vlm_model import VisionLanguageModel
from utils.video_processing import VideoProcessor

app = FastAPI(title="Video Q&A System", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3006", "http://localhost:3000", "http://localhost:3007"],  # all frontend ports you use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
whisper_model = None
vlm_model = None
video_processor = VideoProcessor()

# In-memory storage for current session
current_session = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global whisper_model, vlm_model
    print("Loading models...")
    whisper_model = WhisperTranscriber()
    vlm_model = VisionLanguageModel()
    print("Models loaded successfully!")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process a 5-second video clip"""
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        upload_dir = Path("uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video
        processed_data = await process_video(file_path, session_id)
        
        return {
            "session_id": session_id,
            "message": "Video uploaded and processed successfully",
            "transcript": processed_data["transcript"],
            "duration": processed_data["duration"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/ask-question")
async def ask_question(data: Dict[str, Any]):
    """Process a question about the uploaded video"""
    try:
        session_id = data.get("session_id")
        question = data.get("question")
        
        if not session_id or not question:
            raise HTTPException(status_code=400, detail="Session ID and question required")
        
        if session_id not in current_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get answer from VLM
        answer = await get_answer(session_id, question)
        
        return {
            "question": question,
            "answer": answer,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

async def process_video(file_path: Path, session_id: str) -> Dict[str, Any]:
    """Process uploaded video clip"""
    try:
        # Extract basic video info
        video_info = video_processor.get_video_info(file_path)
        
        # Extract audio and transcribe
        audio_path = video_processor.extract_audio(file_path)
        transcript = whisper_model.transcribe(audio_path)
        
        # Extract key frames
        frames = video_processor.extract_frames(file_path, num_frames=5)
        
        # Store session data
        current_session[session_id] = {
            "video_path": file_path,
            "transcript": transcript,
            "frames": frames,
            "video_info": video_info
        }
        
        return {
            "transcript": transcript,
            "duration": video_info["duration"],
            "frames_count": len(frames)
        }
        
    except Exception as e:
        raise Exception(f"Video processing failed: {str(e)}")

async def get_answer(session_id: str, question: str) -> str:
    """Get answer from vision-language model"""
    try:
        session_data = current_session[session_id]
        
        # Prepare context for VLM
        context = {
            "video_path": session_data["video_path"],
            "transcript": session_data["transcript"],
            "frames": session_data["frames"],
            "question": question
        }
        
        # Get answer from VLM
        # understand the generate_answer function - how does this completely work?
        answer = vlm_model.generate_answer(context)
        
        return answer
        
    except Exception as e:
        raise Exception(f"Question processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": whisper_model is not None and vlm_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
