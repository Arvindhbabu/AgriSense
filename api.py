# api.py
import os
import io
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google import genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import Orchestrator
from tools.market_tool import MarketTool
from tools.weather_tool import WeatherTool

app = FastAPI(title="AgriSense Agents API", version="2.0.0")


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
    market_tool=MarketTool(),
    weather_tool=WeatherTool(),
    local_model=None,  # configure your trained model here
)


@app.get("/")
def health():
    return {"status": "ok", "service": "agrisense", "version": "1.0.0"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    crop_type: str = Form("wheat"),
    user_id: str = Form("farmer_api"),
    location: str = Form(""),
    market_region: str = Form(""),
):
    """
    POST /analyze
    form-data:
      file: image file
      crop_type: optional crop name
      user_id: optional user id
    """
    try:
        content = await file.read(int(os.getenv("AGRISENSE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))) + 1)
        if not content or len(content) > int(os.getenv("AGRISENSE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))):
            return JSONResponse(status_code=413, content={"error": "Image is empty or exceeds the upload limit."})
        Image.open(io.BytesIO(content)).verify()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file provided."},
        )

    try:
        result = await asyncio.to_thread(
            orch.run_case,
            user_id=user_id.strip()[:120] or "farmer_api",
            image_input=io.BytesIO(content),
            crop_type=crop_type,
            config={
                "location": location.strip()[:200] or None,
                "market_region": market_region.strip()[:100] or None,
            },
        )
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
