"""Multi-backend crop diagnosis agent.

Preferred order:
- configured local model first when provided;
- otherwise configured Gemini/MCP integrations before the demo RGB heuristic.

The RGB heuristic is included only as a fallback for wiring/UI demos. It is not a
validated disease classifier and is detected by the uncertainty gate downstream.
"""
import base64
import io
import json
import logging
import os
import re

import numpy as np
from PIL import Image
from utils.image_quality import assess_image_quality

logger = logging.getLogger("agrisense.diagnosis")
logger.setLevel(logging.INFO)


class DiagnosisAgent:
    def __init__(
        self,
        mcp_tool=None,
        gemini_client=None,
        local_model=None,
        backend_preference=None,
        class_names=None,
    ):
        self.mcp = mcp_tool
        self.gemini = gemini_client
        self.local_model = local_model
        self.class_names = class_names

        if backend_preference:
            self.backends = list(backend_preference)
        elif local_model is not None:
            self.backends = ["local", "gemini", "mcp"]
        else:
            # Do not let the always-returning RGB demo heuristic shadow real configured backends.
            self.backends = ["gemini", "mcp", "local"]

    def _load_image(self, image_path_or_bytes):
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            source = io.BytesIO(image_path_or_bytes)
        elif hasattr(image_path_or_bytes, "read"):
            source = image_path_or_bytes
            try:
                source.seek(0)
            except (AttributeError, OSError):
                pass
        elif isinstance(image_path_or_bytes, (str, os.PathLike)):
            source = image_path_or_bytes
        else:
            raise ValueError("Unsupported image input type")
        try:
            return Image.open(source).convert("RGB")
        except Exception as exc:
            raise ValueError("Could not decode the uploaded image") from exc

    def _preprocess_pil(self, image, target=(224, 224)):
        resized = image.resize(target)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _format_prediction(self, raw_pred):
        """Normalize common prediction formats to a JSON-friendly mapping."""
        if isinstance(raw_pred, dict) and "label" in raw_pred and "confidence" in raw_pred:
            result = dict(raw_pred)
            try:
                result["confidence"] = float(result["confidence"])
            except (TypeError, ValueError):
                result["confidence"] = 0.0
            return result

        if isinstance(raw_pred, str):
            text = raw_pred.strip()
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict) and "label" in parsed and "confidence" in parsed:
                        return self._format_prediction(parsed)
                except (ValueError, TypeError):
                    pass

        try:
            arr = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
            if arr.size:
                idx = int(np.argmax(arr))
                confidence = float(arr[idx])
                # Convert likely logits to probabilities; keep valid probability vectors unchanged.
                if np.any(arr < 0) or not np.isclose(float(arr.sum()), 1.0, atol=0.02):
                    shifted = arr - np.max(arr)
                    exp = np.exp(shifted)
                    probs = exp / max(float(exp.sum()), 1e-12)
                    confidence = float(probs[idx])
                label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else str(idx)
                return {"label": str(label), "confidence": confidence, "raw": arr.tolist()}
        except Exception:
            pass
        return {"label": "unknown", "confidence": 0.0, "raw": str(raw_pred)}

    def _run_local(self, pil_img):
        if self.local_model is not None:
            batch = np.expand_dims(self._preprocess_pil(pil_img), axis=0)

            # Keras/TensorFlow-style API.
            predict_fn = getattr(self.local_model, "predict", None)
            if callable(predict_fn):
                try:
                    predictions = predict_fn(batch, verbose=0)
                except TypeError:
                    predictions = predict_fn(batch)
                result = self._format_prediction(predictions)
                result.setdefault("backend", "local_model")
                result.setdefault("model_version", os.getenv("AGRISENSE_MODEL_VERSION", "local-configured"))
                result.setdefault("confidence_calibrated", False)
                return result

            # PyTorch-style API, imported only when the supplied model needs it.
            if hasattr(self.local_model, "parameters") and hasattr(self.local_model, "eval"):
                try:
                    import torch

                    device = next(self.local_model.parameters()).device
                    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                    self.local_model.eval()
                    with torch.no_grad():
                        output = self.local_model(tensor)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    probabilities = torch.softmax(output, dim=-1).detach().cpu().numpy()
                    result = self._format_prediction(probabilities)
                    result.setdefault("backend", "local_model")
                    result.setdefault("model_version", os.getenv("AGRISENSE_MODEL_VERSION", "local-configured"))
                    result.setdefault("confidence_calibrated", False)
                    return result
                except Exception as exc:
                    raise RuntimeError("PyTorch local-model inference failed: " + str(exc)) from exc

            raise TypeError("local_model must expose a Keras-style predict() or PyTorch-style eval()/parameters API")

        # Deliberately weak fallback for local UI smoke tests only.
        arr = self._preprocess_pil(pil_img)
        r_mean = float(arr[..., 0].mean())
        g_mean = float(arr[..., 1].mean())
        b_mean = float(arr[..., 2].mean())
        score = g_mean - (r_mean + b_mean) / 2.0
        conf = float(min(max((score + 0.1) / 0.2, 0.0), 1.0))
        if score > 0.02:
            label, confidence = "healthy", conf
        else:
            label, confidence = "suspect_spot", 1.0 - conf
        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "backend": "demo_rgb_heuristic",
            "raw": {"r": r_mean, "g": g_mean, "b": b_mean},
            "confidence_calibrated": False,
        }

    def _run_mcp(self, pil_img):
        if self.mcp is None:
            raise RuntimeError("MCP backend selected but no MCP client provided")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        payload = {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
        tried = []
        for tool_name in ("classify_image", "predict_image", "getTinyImage"):
            try:
                tried.append(tool_name)
                result = self.mcp.call(tool_name=tool_name, payload=payload)
                normalized = self._format_prediction(result)
                normalized.setdefault("backend", "mcp")
                normalized.setdefault("confidence_calibrated", False)
                return normalized
            except Exception as exc:
                logger.warning("MCP tool %s failed: %s", tool_name, exc)
        raise RuntimeError("MCP calls failed for tools: " + ", ".join(tried))

    def _run_gemini(self, pil_img, crop_type=None):
        if self.gemini is None:
            raise RuntimeError("Gemini backend selected but GEMINI_API_KEY is not configured.")

        model_name = (
            os.getenv("GEMINI_VISION_MODEL")
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.8-flash"
        )
        prompt = (
            "You are assisting with preliminary crop-health triage. Inspect the provided image. "
            "Do not claim certainty when visual evidence is insufficient. If the image is not a crop/leaf "
            "or the disease cannot be distinguished, use label 'unknown' and low confidence. "
            "Return only valid JSON with keys: label (short snake_case class), confidence (number 0 to 1, "
            "your visual estimate, not a calibrated probability), crop (best effort), notes (brief visual signs), "
            "and needs_expert_review (boolean). Do not prescribe pesticide dosage. "
            "User-provided crop type: " + str(crop_type or "not specified") + "."
        )

        models = getattr(self.gemini, "models", None)
        generate = getattr(models, "generate_content", None)
        if not callable(generate):
            raise RuntimeError("Gemini client has no models.generate_content method.")

        response = generate(model=model_name, contents=[prompt, pil_img])
        response_text = getattr(response, "text", None)
        if not response_text:
            raise RuntimeError("Gemini returned an empty text response.")

        match = re.search(r"\{.*\}", response_text, re.S)
        if not match:
            raise RuntimeError("Gemini response was not valid JSON.")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini response contained malformed JSON.") from exc

        if not isinstance(payload, dict) or not payload.get("label"):
            raise RuntimeError("Gemini response did not include a diagnosis label.")

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        raw_review = payload.get("needs_expert_review", False)
        needs_review = raw_review is True or str(raw_review).strip().lower() in {"true", "yes", "1"}
        return {
            "label": str(payload.get("label", "unknown")).strip().lower().replace(" ", "_"),
            "confidence": confidence,
            "crop": payload.get("crop") or crop_type,
            "notes": str(payload.get("notes", "")),
            "needs_expert_review": needs_review,
            "backend": "gemini",
            "model_version": model_name,
            "confidence_calibrated": False,
        }

    def run(self, image_input, crop_type=None):
        pil_img = self._load_image(image_input)
        image_quality = assess_image_quality(pil_img)
        last_error = None

        for backend in self.backends:
            try:
                normalized_backend = str(backend).lower()
                if normalized_backend.startswith("local"):
                    result = self._run_local(pil_img)
                elif normalized_backend.startswith("mcp"):
                    result = self._run_mcp(pil_img)
                elif normalized_backend.startswith("gemi"):
                    result = self._run_gemini(pil_img, crop_type=crop_type)
                else:
                    raise ValueError("Unknown diagnosis backend: " + str(backend))

                result.setdefault("crop", crop_type)
                result["image_quality"] = image_quality
                if image_quality.get("review_required"):
                    result["needs_expert_review"] = True
                return result
            except Exception as exc:
                logger.warning("Diagnosis backend %s failed: %s", backend, exc)
                last_error = exc

        logger.error("All diagnosis backends failed: %s", last_error)
        return {
            "label": "unknown",
            "confidence": 0.0,
            "crop": crop_type,
            "error": "All configured diagnosis backends failed.",
            "last_backend_error": str(last_error) if last_error else None,
            "image_quality": image_quality,
            "needs_expert_review": bool(image_quality.get("review_required")),
        }
