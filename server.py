#!/usr/bin/env python3
"""
OpenAI-compatible TTS API for Chatterbox with streaming support
"""
import argparse
import io
import os
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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatterbox TTS API Server")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run model on (cuda/cpu)")
parser.add_argument("--audio-prompt", type=str, default=None,
                    help="Path to audio file to use as voice prompt")
parser.add_argument("--chunk-size", type=int, default=4096,
                    help="Chunk size for streaming audio (bytes)")
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

async def generate_audio_stream(wav_tensor: torch.Tensor, sample_rate: int, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    """
    Stream audio data in chunks with WAV header
    """
    # Convert tensor to WAV bytes
    wav_bytes = tensor_to_wav_bytes(wav_tensor, sample_rate)
    
    # Stream the WAV data in chunks
    for i in range(0, len(wav_bytes), chunk_size):
        chunk = wav_bytes[i:i + chunk_size]
        yield chunk
        # Small delay to simulate streaming behavior
        await asyncio.sleep(0.01)

async def generate_pcm_stream(wav_tensor: torch.Tensor, sample_rate: int, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    """
    Stream raw PCM audio data in chunks (no WAV header)
    Useful for real-time streaming applications
    """
    # Ensure tensor is on CPU and convert to int16 PCM format
    audio_np = wav_tensor.cpu().numpy()

    # Handle channel dimension
    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
        # Transpose from (channels, samples) to (samples, channels)
        audio_np = audio_np.T

    # Normalize to int16 range if needed
    if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
        audio_np = (audio_np * 32767).astype(np.int16)

    audio_bytes = audio_np.tobytes()

    # Stream in chunks
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(0.001)  # Minimal delay for real-time feel

@app.on_event("startup")
async def load_model():
    """Load the Chatterbox TTS model on startup"""
    global model
    print(f"Loading Chatterbox TTS model on {args.device}...")
    model = ChatterboxTTS.from_pretrained(device=args.device)
    print("Model loaded successfully!")
    if audio_prompt_path:
        print(f"Using default audio prompt: {audio_prompt_path}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "chatterbox-tts",
        "device": args.device,
        "streaming_enabled": True
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
    OpenAI-compatible TTS endpoint with streaming support
    POST /v1/audio/speech

    Set "stream": true in the request body to enable streaming
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        # Determine which audio prompt to use
        prompt_path = request.audio_prompt_path or audio_prompt_path

        # Generate audio
        if prompt_path:
            wav = model.generate(request.input, audio_prompt_path=prompt_path)
        else:
            wav = model.generate(request.input)

        # Handle streaming vs non-streaming
        if request.stream:
            # Stream the audio in chunks
            return StreamingResponse(
                generate_audio_stream(wav, model.sr, chunk_size=args.chunk_size),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # Return complete audio file
            wav_bytes = tensor_to_wav_bytes(wav, model.sr)
            buffer = io.BytesIO(wav_bytes)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav"
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSRequestAudio):
    """
    Dedicated streaming endpoint (always streams)
    POST /v1/audio/speech/stream
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        prompt_path = request.audio_prompt_path or audio_prompt_path

        if prompt_path:
            wav = model.generate(request.input, audio_prompt_path=prompt_path)
        else:
            wav = model.generate(request.input)

        # Always stream
        return StreamingResponse(
            generate_audio_stream(wav, model.sr, chunk_size=args.chunk_size),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Transfer-Encoding": "chunked",
                "X-Streaming": "true"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/v1/audio/speech/pcm")
async def create_speech_pcm_stream(request: TTSRequestAudio):
    """
    Raw PCM streaming endpoint (no WAV header, for real-time applications)
    POST /v1/audio/speech/pcm

    Returns raw PCM audio data (int16, mono/stereo depending on model output)
    Sample rate can be retrieved from GET / endpoint or is typically 24000 Hz
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        prompt_path = request.audio_prompt_path or audio_prompt_path

        if prompt_path:
            wav = model.generate(request.input, audio_prompt_path=prompt_path)
        else:
            wav = model.generate(request.input)

        # Stream raw PCM
        return StreamingResponse(
            generate_pcm_stream(wav, model.sr, chunk_size=args.chunk_size),
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=speech.pcm",
                "Transfer-Encoding": "chunked",
                "X-Sample-Rate": str(model.sr),
                "X-Bit-Depth": "16",
                "X-Channels": str(wav.shape[0] if wav.dim() > 1 else 1),
                "X-Streaming": "true"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/v1/audio/speech/custom")
async def create_speech_custom_voice(request: TTSRequestAudio):
    """
    Extended endpoint that allows per-request audio prompt override
    POST /v1/audio/speech/custom

    Supports streaming via "stream": true
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    if not request.audio_prompt_path:
        raise HTTPException(status_code=400, detail="audio_prompt_path is required for this endpoint")

    if not Path(request.audio_prompt_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio prompt file not found: {request.audio_prompt_path}")

    try:
        wav = model.generate(request.input, audio_prompt_path=request.audio_prompt_path)

        if request.stream:
            return StreamingResponse(
                generate_audio_stream(wav, model.sr, chunk_size=args.chunk_size),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            wav_bytes = tensor_to_wav_bytes(wav, model.sr)
            buffer = io.BytesIO(wav_bytes)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav"
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
