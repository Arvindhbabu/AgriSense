"""
run.py — AgriSense local runner (VS Code / Terminal)

Features:
- Looks for an image file (defaults to 'leaf1.png' if present, otherwise creates a demo image).
- Loads GEMINI_API_KEY from:
    1) Kaggle secrets (if running on Kaggle)
    2) .env file (if python-dotenv installed)
    3) Environment variable GEMINI_API_KEY
- Safely initializes google.genai.Client (if installed and key present). Falls back gracefully if not.
- Instantiates Orchestrator from your project and runs an end-to-end flow.
- Accepts CLI args --image, --user, --crop
- Prints a readable result.
"""
import os
import sys
import argparse
import json
import traceback

# Optional: load .env if present (useful for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not installed or .env missing — continue
    pass

# Try to support Kaggle secrets if someone runs this inside Kaggle
def load_secret_from_kaggle(key_name: str):
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        val = client.get_secret(key_name)
        if val:
            print(f"[INFO] Loaded {key_name} from Kaggle Secrets.")
        return val
    except Exception:
        return None

def get_gemini_api_key():
    # 1) Kaggle Secrets
    val = load_secret_from_kaggle("GEMINI_API_KEY")
    if val:
        return val
    # 2) Environment variable (or .env loaded)
    val = os.environ.get("GEMINI_API_KEY")
    if val:
        print("[INFO] Loaded GEMINI_API_KEY from environment.")
    return val

# Try to create gemini client if available and key present
def get_gemini_client():
    try:
        # import lazily so script still runs if genai not installed
        from google import genai
    except Exception:
        # genai not installed
        print("[WARN] google.genai module not available. Running without Gemini.")
        return None

    key = get_gemini_api_key()
    if not key:
        print("[WARN] GEMINI_API_KEY not found. Running without Gemini.")
        return None

    try:
        client = genai.Client(api_key=key)
        print("[INFO] Gemini client created.")
        return client
    except Exception as e:
        print("[WARN] Failed to create Gemini client:", e)
        return None

# Ensure current working directory is project root (where run.py is)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Decide which image to use
def ensure_image(path_hint=None):
    # Priority:
    # 1) CLI-provided path_hint
    # 2) leaf1.png in project root
    # 3) test_leaf.jpg (auto-create)
    import pathlib
    if path_hint:
        p = pathlib.Path(path_hint)
        if p.exists():
            return str(p)
        else:
            print(f"[WARN] Provided image '{path_hint}' not found; falling back.")
    p_leaf = pathlib.Path(PROJECT_ROOT) / "leaf1.png"
    if p_leaf.exists():
        return str(p_leaf)
    p_jpg = pathlib.Path(PROJECT_ROOT) / "test_leaf.jpg"
    if p_jpg.exists():
        return str(p_jpg)
    # create demo image
    try:
        from PIL import Image
        demo = Image.new("RGB", (224, 224), (80, 180, 90))
        demo.save(str(p_jpg))
        print("[INFO] Created demo image:", str(p_jpg))
        return str(p_jpg)
    except Exception as e:
        raise RuntimeError("Unable to create demo image; please provide an image file.") from e

# Pretty-print results
def pretty_print_result(result):
    try:
        print("\n" + "="*40 + " AGENTS RESULT " + "="*40)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*100 + "\n")
    except Exception:
        print("Result (raw):", result)

def main():
    parser = argparse.ArgumentParser(description="Run AgriSense orchestrator locally")
    parser.add_argument("--image", "-i", default=None, help="Path to leaf image (e.g., leaf1.png)")
    parser.add_argument("--user", "-u", default="farmer_local", help="User id for session")
    parser.add_argument("--crop", "-c", default="wheat", help="Crop type (optional)")
    args = parser.parse_args()

    # Create gemini client (if possible)
    gemini_client = get_gemini_client()

    # Import orchestrator (allow it to import project modules)
    try:
        # The orchestrator in your project expects arguments gemini_client, mcp_gateway_url, local_model
        from agents.orchestrator import Orchestrator
    except Exception as exc:
        print("[ERROR] Failed to import Orchestrator from agents.orchestrator. Traceback:")
        traceback.print_exc()
        sys.exit(1)

    # Choose image
    try:
        image_path = ensure_image(args.image)
        print(f"[INFO] Using image: {image_path}")
    except Exception as e:
        print("[ERROR] Could not prepare an image:", e)
        traceback.print_exc()
        sys.exit(1)

    # Create orchestrator instance
    # If you have a local model object, load it and pass as local_model=...
    # If you run an MCP gateway, set mcp_gateway_url="http://localhost:3000" or appropriate value
    orchestrator = Orchestrator(
        gemini_client=gemini_client,
        mcp_gateway_url=os.environ.get("MCP_GATEWAY_URL", None),
        local_model=None
    )

    # Run the orchestrator end-to-end
    try:
        result = orchestrator.run_case(
            user_id=args.user,
            image_input=image_path,
            crop_type=args.crop
        )
        pretty_print_result(result)
    except Exception as e:
        print("[ERROR] Orchestrator run failed. Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
