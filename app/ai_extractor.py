from pdf2image import convert_from_path
import pytesseract


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using OCR.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    # Initialize an empty string to hold the extracted text
    extracted_text = ""
    
    # Iterate through each image and extract text using pytesseract
    for image in images:
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
    
    return extracted_text



if __name__ == "__main__":
    pdf_file_path = "/home/skull-walla/Downloads/kisumu_sale_1.pdf"  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_file_path)
    print(text)