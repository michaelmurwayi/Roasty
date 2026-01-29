import json
import logging
import re
import argparse
from typing import Dict

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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 180

# =========================================================
# PROMPTS
# =========================================================

TEXT_CLEANUP_PROMPT = """
You are a document reconstruction engine.

Given raw extracted text from a PDF:
- Remove noise
- Merge broken lines
- Preserve tables in text form
- Preserve headings and sections
- DO NOT summarize
- DO NOT add new information
- Return clean, readable document text ONLY

DOCUMENT:
-----------------
{raw_text}
-----------------
"""

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
                "options": {
                    "temperature": 0
                }
            },
            timeout=OLLAMA_TIMEOUT
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
    logger.info("Cleaning document text via LLM")

    prompt = TEXT_CLEANUP_PROMPT.format(raw_text=raw_text)
    return call_ollama(prompt)

# =========================================================
# JSON EXTRACTION (DEFENSIVE)
# =========================================================

def extract_json(llm_output: str) -> Dict:
    logger.info("Extracting JSON from LLM output")

    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    if not match:
        logger.error("No JSON found in LLM output")
        raise ValueError("Invalid LLM output: JSON not found")

    json_str = match.group(0)
    json_str = json_str.replace("None", "null")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.exception("JSON parsing failed")
        raise ValueError("Invalid JSON structure") from e

# =========================================================
# DATA EXTRACTION
# =========================================================

def extract_sale_data(clean_text: str) -> Dict:
    logger.info("Extracting structured sale data")

    prompt = DATA_EXTRACTION_PROMPT.format(
        document_text=clean_text
    )

    llm_output = call_ollama(prompt)
    return extract_json(llm_output)

# =========================================================
# SCHEMA VALIDATION
# =========================================================

def validate_sale_data(data: Dict) -> ExtractedSaleData:
    logger.info("Validating extracted data against schema")

    try:
        return ExtractedSaleData(**data)
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
    structured_data = extract_sale_data(clean_text)
    validated_data = validate_sale_data(structured_data)

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