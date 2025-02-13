from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
import os
import uvicorn

# Create FastAPI instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
VOICE = "en-GB-ThomasNeural"
OUTPUT_FILE = "output.wav"

@app.get("/")
def read_root():
    return {"message": "TTS Server is running"}

@app.get("/tts/")    # Note the trailing slash
async def generate_voice(
    text: str = Query(default="test", description="Text to convert to speech")
):
    try:
        # Create TTS
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(OUTPUT_FILE, "wav")
        
        # Return file
        return FileResponse(OUTPUT_FILE, media_type='audio/wav')
    except Exception as e:
        # Clean up and raise error
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("edgetts:app", host="0.0.0.0", port=8000, reload=True)
