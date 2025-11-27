# api.py
import os
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google import genai
from PIL import Image

from agents.orchestrator import Orchestrator

app = FastAPI(title="AgriSense Agents API", version="1.0.0")


def create_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not set; running without Gemini.")
        return None
    client = genai.Client(api_key=api_key)
    print("[INFO] Gemini client initialized.")
    return client


# Instantiate orchestrator once per process
gemini_client = create_gemini_client()
orch = Orchestrator(
    gemini_client=gemini_client,
    mcp_gateway_url=os.environ.get("MCP_GATEWAY_URL"),
    local_model=None,  # replace with your model if you want
)


@app.get("/")
def health():
    return {"status": "ok", "service": "agrisense", "version": "1.0.0"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    crop_type: str = Form("wheat"),
    user_id: str = Form("farmer_api"),
):
    """
    POST /analyze
    form-data:
      file: image file
      crop_type: optional crop name
      user_id: optional user id
    """
    try:
        content = await file.read()
        # optional: validate image
        Image.open(io.BytesIO(content)).verify()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file provided."},
        )

    try:
        result = orch.run_case(
            user_id=user_id,
            image_input=io.BytesIO(content),
            crop_type=crop_type,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
