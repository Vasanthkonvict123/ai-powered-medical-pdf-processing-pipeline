
"""
ai-powered-medical-pdf-processing-pipeline
===================================
A generic AI-powered system for extracting structured data from medical documents.

DISCLAIMER: This is a sanitized version for educational/portfolio purposes.
Contains no real patient data, company information, or proprietary algorithms.
Designed to demonstrate PDF processing, OCR, and AI extraction techniques.

HIPAA Compliance Note: This code processes documents but stores no PHI.
All examples use fictitious data. Implement proper security measures for production use.

"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2
import openpyxl
from openpyxl import Workbook
import google.generativeai as genai
import shutil
import zipfile
import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter

# ==================== CONFIGURATION ====================

# Base directory configuration - use environment variables in production
BASE_DIR = Path(os.getenv("PROJECT_BASE_DIR", "./document_processor"))
FOLDER_FULL_TEXT = BASE_DIR / "01_Full_Text"
FOLDER_SEPARATE_TEXT = BASE_DIR / "02_Separated_Text"
FOLDER_SEPARATE_PDF = BASE_DIR / "03_Separated_PDFs"
FOLDER_EXCEL_OUTPUT = BASE_DIR / "04_Excel_Output"
FOLDER_CATEGORIZED = BASE_DIR / "05_Categorized_Docs"
FOLDER_IMAGES = BASE_DIR / "06_Cropped_Images"
FOLDER_ARCHIVE = BASE_DIR / "Archive"
FOLDER_INPUT = BASE_DIR / "Input"

# API Configuration - ALWAYS use environment variables in production
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

GEMINI_MODEL = "gemini-2.0-flash-exp"

# =========================== DOCUMENT CLASSIFICATION KEYWORDS ===========================

# Generic keywords for document type identification
NEW_DOCUMENT_KEYWORDS = [
    "MEDICAL FACILITY", "HEALTHCARE PROVIDER",
    "PATIENT AGREEMENT", "MEDICAL RECORD"
]

PRESCRIPTION_KEYWORDS = [
    "PRESCRIPTION", "MEDICATION ORDER",
    "PRESCRIBER", "RX"
]

DEMOGRAPHIC_KEYWORDS = [
    "PATIENT INFORMATION", "DEMOGRAPHICS",
    "PATIENT NAME", "DATE OF BIRTH"
]

INSURANCE_KEYWORDS = [
    "INSURANCE", "COVERAGE",
    "POLICY NUMBER", "SUBSCRIBER"
]

# Bounding box coordinates for image regions (example values)
BBOX_CHECKBOX = (400, 450, 525, 500)
BBOX_LOCATION = (120, 150, 270, 200)

# ==================== AI PROMPTS (Simplified for Public Release) ====================

CHECKBOX_PROMPT = """Analyze this checkbox image and determine the selection.

Look for checkboxes with labels like: "Right", "Left", "Bilateral"

Rules:
- If you see a mark/check INSIDE a box → that option is selected
- Ignore circles or highlighting AROUND text
- Only marks INSIDE boxes count as selections

Respond with ONLY ONE WORD:
- "Right" if Right is checked
- "Left" if Left is checked  
- "Bilateral" if Bilateral is checked OR both Right and Left are checked
- "NoValue" if nothing is clearly checked

Answer:"""

LOCATION_PROMPT = """Extract the location/facility code from this image.

Look for location identifiers or facility codes.
Return the exact code you find, or "NoValue" if none found.

Answer:"""

# Simplified extraction prompt (removing sensitive medical codes)
EXTRACTION_PROMPT = """
Extract structured information from this medical document text.

Return ONLY valid JSON with these fields:
{{
  "File_Name": "NoValue",
  "First_Name": "NoValue",
  "Last_Name": "NoValue",
  "DOB": "NoValue",
  "Sex": "NoValue",
  "Address": "NoValue",
  "City": "NoValue",
  "State": "NoValue",
  "ZIP": "NoValue",
  "Phone_Number": "NoValue",
  "Email": "NoValue",
  "Insurance_Name": "NoValue",
  "Insurance_ID": "NoValue",
  "Location": "NoValue",
  "Document_Date": "NoValue",
  "Provider_Name": "NoValue",
  "Diagnosis_Code": "NoValue"
}}

CRITICAL RULES:
1. Return ONLY JSON, no markdown, no explanations
2. Use "NoValue" for missing fields
3. Format dates as MM/DD/YYYY
4. Format phone as (XXX) XXX-XXXX
5. Remove any PII that looks like placeholder data

Input Text:
\"\"\"{input_data}\"\"\"
"""

# ==================== IMAGE ENHANCEMENT FUNCTIONS ====================

def enhance_image_minimal(img_cv, enhancement_type="general"):
    """
    Minimal image enhancement for better OCR.
    
    Args:
        img_cv: OpenCV image array
        enhancement_type: "checkbox", "text", or "general"
    """
    scale_factor = 1.5
    img_cv = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Light denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=5, 
                                        templateWindowSize=7, 
                                        searchWindowSize=21)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Light sharpening
    kernel = np.array([[0, -0.5, 0], 
                       [-0.5, 3, -0.5], 
                       [0, -0.5, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)
    
    enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr

def enhance_image_for_ocr(image_path: str, output_path: str = None, 
                          image_type: str = "general") -> str:
    """
    Enhance image quality for better OCR/analysis.
    
    Args:
        image_path: Input image path
        output_path: Output path (overwrites input if None)
        image_type: Type of enhancement to apply
    """
    try:
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        enhanced = enhance_image_minimal(img_cv, image_type)
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        if output_path is None:
            output_path = image_path
        
        enhanced_pil.save(output_path, quality=95, dpi=(300, 300))
        print(f"  ✓ Image enhanced: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ⚠ Error enhancing image: {e}")
        return image_path

def crop_and_enhance_region(pdf_path: str, page_num: int, bbox: tuple, 
                           output_path: str, image_type: str = "general") -> bool:
    """
    Crop a region from PDF page and enhance it.
    
    Args:
        pdf_path: Source PDF path
        page_num: Page number (0-indexed)
        bbox: Bounding box (x0, y0, x1, y1)
        output_path: Where to save cropped image
        image_type: Enhancement type
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            print(f"  ✗ Page {page_num + 1} not found")
            doc.close()
            return False
        
        page = doc[page_num]
        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(clip=rect, dpi=300)
        
        temp_path = output_path.replace('.png', '_temp.png')
        pix.save(temp_path)
        doc.close()
        
        enhance_image_for_ocr(temp_path, output_path, image_type)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error cropping: {e}")
        return False

# ==================== AI ANALYSIS FUNCTIONS ====================

def analyze_image_with_ai(image_path: str, prompt: str) -> str:
    """
    Analyze an image using AI vision model.
    
    Args:
        image_path: Path to image
        prompt: Analysis prompt
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        image = Image.open(io.BytesIO(image_data))
        response = model.generate_content([prompt, image])
        result = response.text.strip()
        
        # Clean response
        result = result.replace('```', '').replace('**', '').strip()
        
        return result
        
    except Exception as e:
        print(f"  ✗ AI analysis error: {e}")
        return "NoValue"

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from all pages in PDF using AI.
    
    Returns:
        List of (page_number, text) tuples
    """
    pages_text = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        print(f"📄 Processing {total_pages} pages")
        
        uploaded_file = genai.upload_file(pdf_path)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        for page_num in range(1, total_pages + 1):
            prompt = f"""Extract all text from page {page_num}.
            
Keep original formatting and line breaks.
For checkboxes: [x] for checked, [ ] for unchecked.
Return only the extracted text, no explanations."""
            
            response = model.generate_content([uploaded_file, prompt])
            text = response.text.strip()
            
            pages_text.append((page_num, text))
            print(f"  ✓ Page {page_num}/{total_pages}")
        
        genai.delete_file(uploaded_file.name)
        return pages_text
        
    except Exception as e:
        print(f"✗ Text extraction error: {e}")
        return []

def extract_structured_data(text: str, file_name: str) -> Dict:
    """
    Extract structured data from text using AI.
    
    Args:
        text: Document text
        file_name: Source file name
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = EXTRACTION_PROMPT.format(input_data=text)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON response
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        data = json.loads(response_text)
        data["File_Name"] = file_name
        
        print(f"  ✓ Extracted: {data.get('First_Name', 'Unknown')} {data.get('Last_Name', 'Unknown')}")
        return data
        
    except Exception as e:
        print(f"  ✗ Extraction error: {e}")
        return create_empty_record(file_name)

def create_empty_record(file_name: str) -> Dict:
    """Create empty data record."""
    return {
        "File_Name": file_name,
        "First_Name": "NoValue",
        "Last_Name": "NoValue",
        "DOB": "NoValue",
        "Sex": "NoValue",
        "Address": "NoValue",
        "City": "NoValue",
        "State": "NoValue",
        "ZIP": "NoValue",
        "Phone_Number": "NoValue",
        "Email": "NoValue",
        "Insurance_Name": "NoValue",
        "Insurance_ID": "NoValue",
        "Location": "NoValue",
        "Document_Date": "NoValue",
        "Provider_Name": "NoValue",
        "Diagnosis_Code": "NoValue"
    }

# ==================== DOCUMENT CLASSIFICATION ====================

def is_new_document(text: str) -> bool:
    """Check if text indicates start of new document."""
    text_upper = text.upper()
    matches = sum(1 for kw in NEW_DOCUMENT_KEYWORDS if kw in text_upper)
    return matches >= 2

def classify_page_type(text: str) -> str:
    """
    Classify page type.
    
    Returns:
        "prescription", "demographics", "insurance", or "other"
    """
    text_upper = text.upper()
    
    rx_score = sum(1 for kw in PRESCRIPTION_KEYWORDS if kw in text_upper)
    demo_score = sum(1 for kw in DEMOGRAPHIC_KEYWORDS if kw in text_upper)
    ins_score = sum(1 for kw in INSURANCE_KEYWORDS if kw in text_upper)
    
    if rx_score >= 2:
        return "prescription"
    elif demo_score >= 2:
        return "demographics"
    elif ins_score >= 1:
        return "insurance"
    else:
        return "other"

# ==================== FILE OPERATIONS ====================

def setup_folders():
    """Create all required folders."""
    folders = [
        FOLDER_FULL_TEXT, FOLDER_SEPARATE_TEXT, FOLDER_SEPARATE_PDF,
        FOLDER_EXCEL_OUTPUT, FOLDER_CATEGORIZED, FOLDER_IMAGES,
        FOLDER_ARCHIVE, FOLDER_INPUT
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    print("✓ Folder structure created")

def save_combined_text(pages_text: List[Tuple[int, str]], output_path: str):
    """Save all extracted text to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Document Text Export\n")
            f.write(f"Total Pages: {len(pages_text)}\n")
            f.write(f"{'='*60}\n\n")
            
            for page_num, text in pages_text:
                f.write(f"{'='*60}\n")
                f.write(f"PAGE {page_num}\n")
                f.write(f"{'='*60}\n")
                f.write(text)
                f.write("\n\n")
        
        print(f"✓ Text saved: {output_path}")
        
    except Exception as e:
        print(f"✗ Save error: {e}")

def save_to_excel(data_list: List[Dict], excel_path: str):
    """Save extracted data to Excel."""
    try:
        wb = Workbook()
        ws = wb.active
        
        if data_list:
            headers = list(data_list[0].keys())
            ws.append(headers)
            
            for data in data_list:
                row = [data.get(key, "NoValue") for key in headers]
                ws.append(row)
        
        wb.save(excel_path)
        print(f"✓ Excel saved: {excel_path}")
        print(f"  Records: {len(data_list)}")
        
    except Exception as e:
        print(f"✗ Excel error: {e}")

def split_pdf_pages(pdf_path: str, page_numbers: List[int], output_path: str):
    """Extract specific pages to new PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()
            
            for page_num in page_numbers:
                pdf_writer.add_page(pdf_reader.pages[page_num - 1])
            
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        
        print(f"  ✓ PDF saved: {output_path}")
        
    except Exception as e:
        print(f"  ✗ PDF split error: {e}")

# ==================== MAIN PROCESSING ====================

def process_document(pdf_path: str):
    """
    Main document processing pipeline.
    
    Args:
        pdf_path: Path to input PDF
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(pdf_path)}")
    print(f"{'='*60}\n")
    
    base_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Extract text from all pages
    print("Step 1: Extracting text...")
    pages_text = extract_text_from_pdf(pdf_path)
    
    if not pages_text:
        print("✗ No text extracted")
        return
    
    # Step 2: Save combined text
    print("\nStep 2: Saving text...")
    text_path = FOLDER_FULL_TEXT / f"{base_name}_{timestamp}.txt"
    save_combined_text(pages_text, str(text_path))
    
    # Step 3: Split into separate documents
    print("\nStep 3: Splitting documents...")
    documents = []
    current_doc = []
    
    for page_num, text in pages_text:
        if is_new_document(text) and current_doc:
            documents.append(current_doc)
            current_doc = [page_num]
        else:
            current_doc.append(page_num)
    
    if current_doc:
        documents.append(current_doc)
    
    print(f"  ✓ Found {len(documents)} document(s)")
    
    # Step 4: Process each document
    print("\nStep 4: Extracting data...")
    all_data = []
    
    for idx, doc_pages in enumerate(documents, 1):
        print(f"\n  Document {idx}:")
        
        doc_name = f"{base_name}_{timestamp}_doc{idx}"
        doc_pdf = FOLDER_SEPARATE_PDF / f"{doc_name}.pdf"
        
        # Save document PDF
        split_pdf_pages(pdf_path, doc_pages, str(doc_pdf))
        
        # Get text for this document
        doc_text = "\n\n".join([text for num, text in pages_text if num in doc_pages])
        
        # Extract structured data
        data = extract_structured_data(doc_text, doc_name)
        data["Source_PDF"] = str(doc_pdf)
        
        all_data.append(data)
    
    # Step 5: Save to Excel
    print("\nStep 5: Saving to Excel...")
    excel_path = FOLDER_EXCEL_OUTPUT / f"extracted_data_{timestamp}.xlsx"
    save_to_excel(all_data, str(excel_path))
    
    print(f"\n{'='*60}")
    print(f"✓ PROCESSING COMPLETE")
    print(f"{'='*60}\n")
    print(f"Processed: {len(documents)} document(s)")
    print(f"Output: {excel_path}\n")

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Medical Document Processing System")
    print("="*60 + "\n")
    
    # Setup
    setup_folders()
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Find PDFs in input folder
    pdf_files = list(FOLDER_INPUT.glob("*.pdf"))
    
    if not pdf_files:
        print(f"✗ No PDFs found in: {FOLDER_INPUT}")
        print("Please add PDF files to process.")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process:\n")
    for idx, pdf in enumerate(pdf_files, 1):
        print(f"  {idx}. {pdf.name}")
    print()
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            process_document(str(pdf_path))
        except Exception as e:
            print(f"\n✗ Error processing {pdf_path.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
