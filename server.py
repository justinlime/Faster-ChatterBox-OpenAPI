#!/usr/bin/env python3
"""
OpenAI-compatible TTS API for Chatterbox with streaming support
Optimized for minimal first-chunk latency
"""
import argparse
import io
import os
import gc
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import numpy as np

# Parse command line arguments with ENV var fallbacks
parser = argparse.ArgumentParser(description="Chatterbox TTS API Server")
parser.add_argument("--port", type=int,
                    default=int(os.getenv("PORT", "8000")),
                    help="Port to run the server on (ENV: PORT)")
parser.add_argument("--host", type=str,
                    default=os.getenv("HOST", "0.0.0.0"),
                    help="Host to bind to (ENV: HOST)")
parser.add_argument("--device", type=str,
                    default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
                    help="Device to run model on (ENV: DEVICE)")
parser.add_argument("--audio-prompt", type=str,
                    default=os.getenv("AUDIO_PROMPT"),
                    help="Path to audio file to use as voice prompt (ENV: AUDIO_PROMPT)")
parser.add_argument("--chunk-size", type=int,
                    default=int(os.getenv("CHUNK_SIZE", "4096")),
                    help="Chunk size for streaming audio in bytes (ENV: CHUNK_SIZE)")
args = parser.parse_args()

# Initialize FastAPI app
app = FastAPI(title="Chatterbox TTS API", version="1.0.0")

# Global model instance
model: Optional[ChatterboxTTS] = None
audio_prompt_path: Optional[str] = args.audio_prompt

class TTSRequest(BaseModel):
    model: str = "tts-1"  # OpenAI compatibility
    input: str
    voice: str = "alloy"  # OpenAI compatibility, ignored
    response_format: str = "wav"
    speed: float = 1.0

class TTSRequestAudio(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0
    audio_prompt_path: Optional[str] = None
    stream: bool = False  # Enable streaming

def tensor_to_wav_bytes(wav_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """
    Convert a torch tensor to WAV format bytes using soundfile
    """
    # Move to CPU and convert to numpy
    audio_np = wav_tensor.cpu().numpy()

    # Handle channel dimension - soundfile expects (samples,) or (samples, channels)
    if audio_np.ndim == 1:
        # Mono audio
        pass
    elif audio_np.ndim == 2:
        # If shape is (channels, samples), transpose to (samples, channels)
        if audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T

    # Create BytesIO buffer
    buffer = io.BytesIO()

    # Write WAV data to buffer using soundfile
    sf.write(buffer, audio_np, sample_rate, format='WAV')

    # Reset buffer position to beginning
    buffer.seek(0)

    return buffer.read()

async def generate_audio_stream_fast(
    text: str, 
    model: ChatterboxTTS, 
    prompt_path: Optional[str], 
    chunk_size: int
) -> AsyncGenerator[bytes, None]:
    """
    Generate and stream audio with minimal latency for first chunk.
    Runs generation in executor to avoid blocking event loop.
    """
    loop = asyncio.get_event_loop()
    
    # Define generation function
    def generate():
        with torch.no_grad():
            if prompt_path:
                return model.generate(text, audio_prompt_path=prompt_path)
            else:
                return model.generate(text)
    
    # Generate audio in thread pool to not block event loop
    wav = await loop.run_in_executor(None, generate)
    
    # Convert to WAV bytes
    wav_bytes = tensor_to_wav_bytes(wav, model.sr)
    
    # Stream chunks immediately without artificial delays
    for i in range(0, len(wav_bytes), chunk_size):
        chunk = wav_bytes[i:i + chunk_size]
        yield chunk

async def generate_audio_stream_incremental(
    text: str, 
    model: ChatterboxTTS, 
    prompt_path: Optional[str], 
    chunk_size: int
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio as it's generated (if model supports streaming).
    Falls back to fast generation if streaming not available.
    """
    sample_rate = model.sr
    
    with torch.no_grad():
        # Check if model has streaming generation capability
        if hasattr(model, 'generate_stream'):
            # True streaming - generate and send chunks as they're created
            async for audio_chunk in model.generate_stream(text, audio_prompt_path=prompt_path):
                wav_bytes = tensor_to_wav_bytes(audio_chunk, sample_rate)
                yield wav_bytes
        else:
            # Fallback to fast generation with async executor
            loop = asyncio.get_event_loop()
            
            def generate():
                if prompt_path:
                    return model.generate(text, audio_prompt_path=prompt_path)
                else:
                    return model.generate(text)
            
            wav = await loop.run_in_executor(None, generate)
            wav_bytes = tensor_to_wav_bytes(wav, sample_rate)
            
            # Stream in chunks
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i:i + chunk_size]

@app.on_event("startup")
async def load_model():
    """
    Load the Chatterbox TTS model on startup with optimizations for fast inference
    """
    global model
    print(f"Loading Chatterbox TTS model on {args.device}...")
    model = ChatterboxTTS.from_pretrained(device=args.device)
    model.t3.to(dtype=torch.float16)
    model.conds.t3.to(dtype=torch.float16)
    
    
    # Warmup the model with a dummy input to avoid slow first inference
    print("Warming up model...")
    try:
        with torch.no_grad():
            warmup_text = "Hello, this is a warmup."
            if audio_prompt_path:
                _ = model.generate(warmup_text, audio_prompt_path=audio_prompt_path)
            else:
                _ = model.generate(warmup_text)
        
        # Clear CUDA cache after warmup to free memory
        if args.device == "cuda":
            torch.cuda.empty_cache()
        
        print("Model warmup complete!")
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")
    
    if audio_prompt_path:
        print(f"Using default audio prompt: {audio_prompt_path}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "chatterbox-tts",
        "device": args.device,
        "streaming_enabled": True,
        "optimized": True
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "chatterbox"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1677610602,
                "owned_by": "chatterbox"
            }
        ]
    }

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequestAudio):
    """
    OpenAI-compatible TTS endpoint with optimized streaming support
    POST /v1/audio/speech

    Set "stream": true in the request body to enable streaming
    
    Optimizations applied:
    - Async executor to avoid blocking event loop
    - No artificial delays in streaming
    - Model warmup on startup
    - Eval mode for faster inference
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        # Determine which audio prompt to use
        prompt_path = request.audio_prompt_path or audio_prompt_path

        # Handle streaming vs non-streaming
        if request.stream:
            # Use optimized streaming with async executor
            response = StreamingResponse(
                generate_audio_stream_fast(
                    request.input,
                    model,
                    prompt_path,
                    args.chunk_size
                ),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                }
            )
        else:
            # Non-streaming: generate in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def generate():
                with torch.no_grad():
                    if prompt_path:
                        return model.generate(request.input, audio_prompt_path=prompt_path)
                    else:
                        return model.generate(request.input)
            
            wav = await loop.run_in_executor(None, generate)
            wav_bytes = tensor_to_wav_bytes(wav, model.sr)
            buffer = io.BytesIO(wav_bytes)

            response = StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav"
                }
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    finally:
        # Optional: Clear CUDA cache after each generation to prevent memory buildup
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
