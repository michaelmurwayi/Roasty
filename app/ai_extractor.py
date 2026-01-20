from pdf2image import convert_from_path
import pytesseract
from schema import ExtractedSaleData
import json
import requests
import re

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"

# Maximum characters per chunk to avoid 500 errors
MAX_CHUNK_SIZE = 2000


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using OCR.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
    return extracted_text


def clean_llm_json(text: str) -> str:
    """
    Extract only the JSON list from LLM output, removing extra text, code blocks, or comments.
    """
    # Remove code blocks
    text = text.replace("```", "")
    # Remove inline comments (e.g., # in kgs)
    text = re.sub(r"#.*", "", text)
    # Extract first JSON list
    match = re.search(r"\[\s*(\{.*?\})\s*(,\s*\{.*?\}\s*)*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON list found in LLM output:\n{text}")
    return match.group(0).strip()


def run_llm_extraction(system_prompt: str, user_prompt_template: str, text: str):
    """
    Safely run LLM extraction on large PDF text by chunking.

    Args:
        system_prompt (str): Instructions for the AI assistant.
        user_prompt_template (str): Template including schema and text.
        text (str): Full extracted PDF text.

    Returns:
        List[ExtractedSaleData]: List of structured extracted sale data.
    """
    # Split text into chunks
    chunks = [text[i:i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
    results = []

    for idx, chunk in enumerate(chunks):
        prompt = user_prompt_template.format(ExtractedSaleData=ExtractedSaleData, text=chunk)
        payload = {
            "model": MODEL,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False
        }
        print(f"Processing chunk {idx + 1}/{len(chunks)}...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()

        # Parse Ollama wrapper JSON
        outer = response.json()
        llm_text = outer.get("response", "")
        if not llm_text:
            raise ValueError(f"No 'response' field found in Ollama output:\n{response.text}")

        # Extract clean JSON list
        json_only = clean_llm_json(llm_text)
        results.append(json_only)

    # Combine all chunk outputs
    final_output = "\n".join(results).strip()

    try:
        data_list = json.loads(final_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nOutput:\n{final_output}")

    # Convert JSON to ExtractedSaleData objects
    return [ExtractedSaleData(**item) for item in data_list]


if __name__ == "__main__":
    pdf_file_path = "/home/skull-walla/Downloads/kisumu_sale_1.pdf"

    SYSTEM_PROMPT = """
    You are an AI assistant that extracts text from PDF.
    Extract Coffee Sale data from noisy PDF text.

    Rules:
    - Normalize spelling mistakes.
    - Weight must be in kgs.
    - Price must be in USD.
    - Single buyer per document.

    Return a valid list of dicts.
    """

    USER_PROMPT = """
    Extract the coffee sale data using this schema:

    {ExtractedSaleData.model_json_schema}

    Text:
    {text}
    """

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file_path)

    # Run extraction
    data_list = run_llm_extraction(SYSTEM_PROMPT, USER_PROMPT, text)

    # Print structured output as JSON
    print(json.dumps([data.dict() for data in data_list], indent=4))
