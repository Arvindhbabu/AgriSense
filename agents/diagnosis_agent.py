# agents/diagnosis_agent.py
"""
DiagnosisAgent: multi-backend image classifier for crop disease detection.

Backends supported:
 - local_model: a TF/PyTorch model loaded in your notebooks (recommended if you trained locally)
 - mcp_tool: MCP server tool (ADK MCP example)
 - gemini_client: Gemini Vision API (cloud)

This corrected version:
 - provides a tiny, safe **local heuristic** classifier when no local model is supplied (great for demos)
 - implements a robust `_run_gemini` template that tries common genai client patterns
 - makes `_run_mcp` resilient and returns a formatted prediction
 - keeps behavior clear so you can replace the heuristic with your real model call later
"""
import os
import io
import base64
import json
import re
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("agrisense.diagnosis")
logger.setLevel(logging.INFO)


class DiagnosisAgent:
    def __init__(self, mcp_tool=None, gemini_client=None, local_model=None, backend_preference=None, class_names=None):
        self.mcp = mcp_tool
        self.gemini = gemini_client
        self.local_model = local_model
        # default preference order
        self.backends = backend_preference or ["local", "mcp", "gemini"]
        self.class_names = class_names  # optional list mapping indices -> labels

    def _load_image(self, image_path_or_bytes):
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
        if hasattr(image_path_or_bytes, "read"):
            return Image.open(image_path_or_bytes).convert("RGB")
        if isinstance(image_path_or_bytes, str):
            return Image.open(image_path_or_bytes).convert("RGB")
        raise ValueError("Unsupported image input type")

    def _preprocess_pil(self, image: Image.Image, target=(224, 224)):
        image = image.resize(target)
        arr = np.array(image).astype("float32") / 255.0
        return arr

    def _format_prediction(self, raw_pred):
        """Normalize several prediction formats to {'label', 'confidence', 'raw'}"""
        if isinstance(raw_pred, dict) and "label" in raw_pred and "confidence" in raw_pred:
            return raw_pred
        # If raw_pred is probability vector-like
        try:
            arr = np.array(raw_pred)
            if arr.ndim >= 1:
                idx = int(np.argmax(arr))
                conf = float(np.max(arr))
                label = (self.class_names[idx] if self.class_names else str(idx))
                return {"label": label, "confidence": conf, "raw": arr.tolist()}
        except Exception:
            pass
        # If raw_pred is JSON string or text containing JSON
        try:
            if isinstance(raw_pred, str):
                m = re.search(r"\{.*\}", raw_pred, re.S)
                if m:
                    parsed = json.loads(m.group(0))
                    if "label" in parsed and "confidence" in parsed:
                        return parsed
        except Exception:
            pass
        return {"label": "unknown", "confidence": 0.0, "raw": str(raw_pred)}

    # -------------------------
    # Local backend (heuristic or real model)
    # -------------------------
    def _run_local(self, pil_img):
        """
        If a local_model object is provided, you should *replace* this body with your model's inference:
            preds = self.local_model.predict(batch)   # TF example
            or use torch for PyTorch models

        If no local_model is supplied this function will run a tiny heuristic classifier
        that looks at green-channel dominance to decide healthy vs suspect_spot.
        This is intentionally simple and suitable for demos only.
        """
        if self.local_model is not None:
            # User must replace these example lines with their model-specific code.
            # Example (TensorFlow):
            # batch = np.expand_dims(self._preprocess_pil(pil_img), 0)
            # preds = self.local_model.predict(batch)
            # return self._format_prediction(preds)
            raise NotImplementedError("Local model provided: paste your model inference code here (TF/PyTorch).")
        # Heuristic fallback (demo)
        try:
            img = pil_img.resize((224, 224))
            arr = np.array(img).astype("float32") / 255.0  # HxWxC
            r_mean = float(arr[..., 0].mean())
            g_mean = float(arr[..., 1].mean())
            b_mean = float(arr[..., 2].mean())
            score = g_mean - (r_mean + b_mean) / 2.0
            # map score to confidence 0..1 (rough)
            conf = float(min(max((score + 0.1) / 0.2, 0.0), 1.0))
            if score > 0.02:
                return {"label": "healthy", "confidence": round(conf, 3), "raw": {"r": r_mean, "g": g_mean, "b": b_mean}}
            else:
                return {"label": "suspect_spot", "confidence": round(1 - conf, 3), "raw": {"r": r_mean, "g": g_mean, "b": b_mean}}
        except Exception as e:
            raise RuntimeError("Local heuristic failed: " + str(e))

    # -------------------------
    # MCP backend
    # -------------------------
    def _run_mcp(self, pil_img):
        """
        Call an MCP tool. Expects self.mcp to implement .call(tool_name, payload).
        Returns normalized prediction dict.
        """
        if self.mcp is None:
            raise RuntimeError("MCP backend selected but no MCP client provided.")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = {"image_base64": b64}
        # Tool names depend on your MCP server; adapt if needed.
        tried = []
        for tool_name in ("classify_image", "predict_image", "getTinyImage"):
            try:
                tried.append(tool_name)
                result = self.mcp.call(tool_name=tool_name, payload=payload)
                logger.info("MCP returned from %s", tool_name)
                return self._format_prediction(result)
            except Exception as e:
                logger.warning("MCP tool %s failed: %s", tool_name, e)
                continue
        raise RuntimeError(f"MCP calls failed for tools: {tried}")

    # -------------------------
    # Gemini backend (robust template)
    # -------------------------
    def _run_gemini(self, pil_img):
        """
        Attempt to call a Gemini / genai client using common patterns.
        If your notebook used a specific call signature, paste that exact snippet here.
        The method tries:
         - gemini.text_generation.create(...)
         - gemini.generate(...)
         - gemini.predict_image / gemini.classify_image wrappers
        It attempts to parse JSON inside returned text if present.
        """
        if self.gemini is None:
            raise RuntimeError("Gemini backend selected but no gemini_client provided.")

        # Convert PIL image to bytes
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        last_exc = None
        # Pattern A: genai.Client.text_generation.create(...)
        try:
            tg = getattr(self.gemini, "text_generation", None)
            if tg is not None and hasattr(tg, "create"):
                prompt = "Identify crop health and likely disease in the image. Reply as JSON: {\"label\":..., \"confidence\":..., \"notes\":...}"
                try:
                    resp = tg.create(model="gemini-1.5", prompt=prompt, image=img_bytes)
                except TypeError:
                    # some clients expect different kwarg names
                    resp = tg.create(model="gemini-1.5", prompt=prompt, content=image_bytes if False else prompt)
                text = getattr(resp, "text", None) or str(resp)
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {"label": text.strip(), "confidence": 0.0, "raw": text}
                return {"label": text.strip(), "confidence": 0.0, "raw": text}
        except Exception as e:
            last_exc = e

        # Pattern B: client.generate(...)
        try:
            if hasattr(self.gemini, "generate"):
                prompt = "Identify crop health and likely disease in the image. Reply as JSON: {\"label\":..., \"confidence\":..., \"notes\":...}"
                try:
                    resp = self.gemini.generate(model="gemini-1.5", input={"image_bytes": img_bytes, "prompt": prompt})
                except TypeError:
                    resp = self.gemini.generate(model="gemini-1.5", prompt=prompt, image=img_bytes)
                text = getattr(resp, "text", None) or str(resp)
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {"label": text.strip(), "confidence": 0.0, "raw": text}
                return {"label": text.strip(), "confidence": 0.0, "raw": text}
        except Exception as e:
            last_exc = e

        # Pattern C: helper methods on client
        for name in ("predict_image", "image_predict", "classify_image"):
            fn = getattr(self.gemini, name, None)
            if callable(fn):
                try:
                    resp = fn(image_bytes=img_bytes)
                    return self._format_prediction(resp)
                except Exception as e:
                    last_exc = e
                    continue

        # If we reached here, we couldn't call Gemini with known patterns
        raise RuntimeError("Gemini client available but call pattern unknown. Last error: " + repr(last_exc))

    # -------------------------
    # Run orchestration of backends
    # -------------------------
    def run(self, image_input, crop_type=None):
        pil = self._load_image(image_input)
        last_exc = None
        for backend in self.backends:
            try:
                b = backend.lower()
                if b.startswith("local"):
                    res = self._run_local(pil)
                elif b.startswith("mcp"):
                    res = self._run_mcp(pil)
                elif b.startswith("gemi") or b.startswith("gemini"):
                    res = self._run_gemini(pil)
                else:
                    raise ValueError("Unknown backend: " + backend)
                res.setdefault("crop", crop_type)
                return res
            except NotImplementedError:
                # Provide a clear message that user must add model code
                logger.exception("Local model inference not implemented. Replace with your model call.")
                last_exc = RuntimeError("Local model inference not implemented.")
                break
            except Exception as e:
                logger.warning("Backend %s failed: %s", backend, e)
                last_exc = e
                continue

        # If none of the backends returned a result, return a safe placeholder
        logger.error("All inference backends failed. Last error: %s", last_exc)
        return {"label": "error", "confidence": 0.0, "error": str(last_exc)}
