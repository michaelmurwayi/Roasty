import json
import logging
import re
import argparse
from typing import Dict, Any

import pdfplumber
import requests
from pydantic import ValidationError

from schema import ExtractedSaleData

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logger = logging.getLogger("ai_extractor")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# =========================================================
# OLLAMA CONFIG
# =========================================================

OLLAMA_URL = "http://192.168.1.213:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 180

# =========================================================
# PROMPTS
# =========================================================



DATA_EXTRACTION_PROMPT = """
Extract buyer and coffee sale details.

Return STRICT JSON ONLY:

{
  "buyer": "string",
  "coffee_details": [
    {
      "seller": "string",
      "outturn": "string",
      "grade": "string",
      "bags": number | null,
      "pockets": number | null,
      "weight_kg": number
    }
  ]
}

Rules:
- Output MUST be valid JSON
- Keys must not contain whitespace or newlines
- Use null instead of None
- No trailing commas
- No markdown
- No explanation
- Use exact values only
- No inference

DOCUMENT:
-----------------
{document_text}
-----------------
"""

# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info("Extracting text from PDF: %s", pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        logger.exception("Failed to extract PDF text")
        raise RuntimeError("PDF extraction failed") from e

# =========================================================
# OLLAMA CLIENT
# =========================================================

def call_ollama(prompt: str) -> str:
    logger.info("Calling Ollama model: %s", OLLAMA_MODEL)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"]

    except requests.RequestException as e:
        logger.exception("Ollama request failed")
        raise RuntimeError("LLM request failed") from e

# =========================================================
# TEXT CLEANUP
# =========================================================

def clean_document_text(raw_text: str) -> str:
    logger.info("Cleaning and extracting structured coffee data using Python")

    text = re.sub(r"\s{2,}", " ", raw_text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    buyer = None
    coffee_details = []

    current_lot = {}

    for line in lines:
        lower = line.lower()

        # -------- BUYER --------
        if buyer is None and "buyer" in lower:
            # e.g. Buyer: Kisumu Limited Liability Company
            buyer = line.split(":", 1)[-1].strip()
            continue

        # -------- SELLER / FARM --------
        if any(k in lower for k in ["fcs", "farm", "estate"]):
            if current_lot:
                coffee_details.append(current_lot)
                current_lot = {}

            current_lot["seller"] = line
            continue

        # -------- OUTTURN --------
        if "outturn" in lower:
            current_lot["outturn"] = line.split()[-1]
            continue

        # -------- GRADE --------
        if "grade" in lower:
            current_lot["grade"] = line.split()[-1]
            continue

        # -------- BAGS --------
        if "bag" in lower:
            match = re.search(r"\d+", line)
            current_lot["bags"] = int(match.group()) if match else None
            continue

        # -------- POCKETS --------
        if "pocket" in lower:
            match = re.search(r"\d+", line)
            current_lot["pockets"] = int(match.group()) if match else None
            continue

        # -------- WEIGHT --------
        if "kg" in lower:
            match = re.search(r"\d+", line)
            if match:
                current_lot["weight_kg"] = int(match.group())
            continue

    if current_lot:
        coffee_details.append(current_lot)

    result = {
        "buyer": buyer,
        "coffee_details": coffee_details
    }

    return json.dumps(result, ensure_ascii=False)


# =========================================================
# JSON NORMALIZATION (FIX)
# =========================================================

def normalize_keys(obj: Any) -> Any:
    """
    Recursively normalize dictionary keys:
    - strip whitespace
    - remove stray quotes
    """
    if isinstance(obj, dict):
        return {
            k.strip().strip('"'): normalize_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    return obj

# =========================================================
# JSON EXTRACTION (DEFENSIVE)
# =========================================================

def extract_json(llm_output: str) -> Dict:
    logger.info("Extracting JSON from LLM output")

    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    if not match:
        logger.error("No JSON found in LLM output")
        raise ValueError("Invalid LLM output: JSON not found")

    json_str = match.group(0).replace("None", "null")

    try:
        data = json.loads(json_str)
        return normalize_keys(data)
    except json.JSONDecodeError as e:
        logger.exception("JSON parsing failed")
        raise ValueError("Invalid JSON structure") from e



# =========================================================
# SCHEMA VALIDATION
# =========================================================

def validate_sale_data(data: Dict) -> ExtractedSaleData:
    logger.info("Validating extracted data against schema")

    try:
        print(data)
        return ExtractedSaleData.model_validate_json(data)
    except ValidationError as e:
        logger.exception("Schema validation failed")
        raise ValueError("Schema validation error") from e

# =========================================================
# PIPELINE ORCHESTRATOR
# =========================================================

def run_extraction_pipeline(pdf_path: str) -> ExtractedSaleData:
    logger.info("Starting extraction pipeline")

    raw_text = extract_text_from_pdf(pdf_path)
    clean_text = clean_document_text(raw_text)
    validated_data = validate_sale_data(clean_text)

    logger.info("Extraction pipeline completed successfully")
    return validated_data

# =========================================================
# MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structured coffee sale data from PDF"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )

    args = parser.parse_args()

    try:
        result = run_extraction_pipeline(args.pdf_path)
        print(json.dumps(result.model_dump(), indent=2))

    except Exception as e:
        logger.error("Extraction failed: %s", e)
        exit(1)
