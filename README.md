# 🏥 AI-Powered Medical PDF Processing Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI: Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)](https://ai.google.dev/)

> An intelligent document processing system that leverages AI vision models to automatically extract, classify, and structure information from complex medical PDF documents.

---

## 🎯 What It Does

Transforms unstructured medical PDFs into clean, structured data automatically:

- 📄 **Reads** multi-patient PDF documents (10s to 100s of pages)
- 🔍 **Identifies** individual patient records automatically  
- 🤖 **Extracts** structured data using AI vision models
- 📑 **Classifies** pages (prescriptions, demographics, insurance, notes)
- 🖼️ **Enhances** images for better OCR and form field detection
- 📊 **Exports** clean data to Excel for further processing

---

## ✨ Key Features

### 🤖 AI-Powered Intelligence
- **Vision Model Integration**: Leverages Google Gemini 2.0 for accurate text extraction
- **Context-Aware Parsing**: Understands document structure and medical terminology
- **Smart Field Detection**: Automatically locates and extracts specific data fields

### 📄 Advanced PDF Processing
- **Multi-Document Splitting**: Detects boundaries and separates records
- **Page Classification**: Categorizes pages by type automatically
- **Rotation Correction**: Fixes page orientation issues

### 🖼️ Computer Vision Enhancement
- **Image Preprocessing**: OpenCV-based enhancement for better OCR
- **Checkbox Detection**: Specialized algorithms for form field recognition
- **Region Extraction**: Crops and analyzes specific document areas

### 📊 Data Management
- **Structured Output**: Clean JSON → Excel conversion
- **Batch Processing**: Handles multiple PDFs in queue
- **Archive System**: Automatic backup with versioning

---

## 🏗️ Architecture

```
Input PDFs
    ↓
AI Text Extraction (Gemini Vision)
    ↓
Document Classification
    ↓
Multi-Document Splitting
    ↓
Image Enhancement (OpenCV)
    ↓
Structured Data Extraction
    ↓
Excel Export
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Google Gemini API key (free tier available)
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-powered-medical-pdf-processing-pipeline.git
cd ai-powered-medical-pdf-processing-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# Set API key (Linux/Mac)
export GEMINI_API_KEY="your_api_key_here"

# Or Windows
set GEMINI_API_KEY=your_api_key_here
```

### Run

```bash
# Place PDFs in Input folder
mkdir -p Input
cp your_documents.pdf Input/

# Run pipeline
python document_processor.py

# Check output
ls Excel_Output/
```

---

## 📦 Requirements

```txt
google-generativeai>=0.3.0
PyPDF2>=3.0.0
openpyxl>=3.1.0
PyMuPDF>=1.23.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
```

Install all at once:

```bash

pip install -r requirements.txt

```

---

## 📊 Sample Workflow

### Input
```

Input/
  └── hospital_records_jan_2024.pdf  (50 pages, 3 patients)

```

### Processing
```bash

$ python document_processor.py

Processing: hospital_records_jan_2024.pdf
  ✓ Extracted 50 pages
  ✓ Found 3 documents
  ✓ Document 1: 18 pages
  ✓ Document 2: 22 pages  
  ✓ Document 3: 10 pages
  ✓ Extracted structured data
  ✓ Excel saved

Processing complete!
```

### Output
```

Excel_Output/

  └── extracted_data_20240115_143052.xlsx

      ├── Record 1: Demographics, Insurance, Clinical data

      ├── Record 2: Demographics, Insurance, Clinical data
      └── Record 3: Demographics, Insurance, Clinical data

Separated_PDFs/
  ├── hospital_records_20240115_doc1.pdf
  ├── hospital_records_20240115_doc2.pdf
  └── hospital_records_20240115_doc3.pdf
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| AI Model | Google Gemini 2.0 Flash | Text extraction & analysis |
| PDF Library | PyMuPDF (fitz) | PDF rendering & manipulation |
| PDF Splitting | PyPDF2 | Document splitting & merging |
| Computer Vision | OpenCV | Image enhancement |
| Image Processing | Pillow (PIL) | Format conversion |
| Data Export | openpyxl | Excel generation |
| Language | Python 3.8+ | Core implementation |

---

## 📈 Performance Metrics

- **Processing Speed**: ~2-3 seconds per page
- **Accuracy**: 90-95% field extraction (clean documents)
- **Scalability**: Handles 100+ page documents
- **Concurrent Processing**: Multiple PDFs in batch

---

## 🔒 Security & Compliance

### ⚠️ Important Disclaimers

- **Educational Purpose**: This is a sanitized, generic implementation
- **No Real Data**: Contains NO actual patient information
- **Not Production-Ready**: Requires security hardening for real-world use

### For Production Use, Implement:

✅ End-to-end encryption  
✅ Audit logging  
✅ Role-based access control  
✅ HIPAA compliance measures  
✅ Data retention policies  
✅ Business Associate Agreements  

---

## 📁 Project Structure

```
ai-powered-medical-pdf-processing-pipeline/
│
├── document_processor.py      # Main processing script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
│
├── Input/                     # Place PDFs here
├── 01_Full_Text/             # Extracted text files
├── 02_Separated_Text/        # Individual document texts
├── 03_Separated_PDFs/        # Split PDF documents
├── 04_Excel_Output/          # Final Excel files
├── 05_Categorized_Docs/      # Docs by type
├── 06_Cropped_Images/        # Enhanced images
└── Archive/                   # Backup archives
```

---

## 🎓 What You'll Learn

This project demonstrates:

✅ **AI/ML Integration** - Working with vision model APIs  
✅ **Computer Vision** - Image preprocessing techniques  
✅ **PDF Processing** - Complex document manipulation  
✅ **Data Pipelines** - ETL workflow design  
✅ **Error Handling** - Robust exception management  
✅ **Code Architecture** - Clean, modular design  

---

## 🐛 Troubleshooting

### Common Issues

**"API Key not found"**
```bash
# Make sure you set the environment variable
export GEMINI_API_KEY="your_key"
```

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"Permission denied"**
```bash
# Check folder permissions
chmod -R 755 ./
```

**Low extraction accuracy**
- Ensure documents are high quality (300 DPI+)
- Check if PDFs are scanned images vs. text-based
- Try enhancing source document quality

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Gemini AI team for providing vision model access
- OpenCV community for computer vision tools
- PyMuPDF developers for excellent PDF library
- Open source community for inspiration

---

## 📧 Contact & Support

**Questions?** Open an issue or reach out:

- 📧 Email: vasanthsoundararajan95@.com

- 💼 LinkedIn: Vasanth S (https://www.linkedin.com/in/vasanthsa/)

---

## ⭐ Star History

If this project helped you, please consider giving it a ⭐️!

---




<p align="center">Made with ❤️ by Vasanth S</p>
