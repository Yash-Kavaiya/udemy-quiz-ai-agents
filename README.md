# 📚 PDF to MCQ Generator 🧠

<div align="center">
  
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

</div>

<p align="center">
  <b>Transform your PDF documents into interactive multiple-choice questions using AI</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Web Interface](#web-interface)
  - [Telegram Bot](#telegram-bot)
- [Technical Details](#-technical-details)
- [Security Considerations](#-security-considerations)
- [License](#-license)

---

## 🔍 Overview

This project provides a solution for automatically generating multiple-choice questions (MCQs) from PDF documents using AI. It extracts text from uploaded PDFs and uses the Google Gemini or Groq AI models to create contextually relevant MCQs complete with options, correct answers, and explanations.

The system offers two interfaces:
- **Web Application**: Upload PDFs and download MCQs directly in your browser
- **Telegram Bot**: Send PDFs via chat and receive MCQs as CSV files

Generated questions are formatted as CSV files compatible with learning management systems that support the Practice Test Bulk Question Upload format.

---

## ✨ Features

- **📄 Intelligent PDF Processing**: Extracts text from PDF files with robust fallback mechanisms
- **🤖 AI-Powered Question Generation**: Creates relevant multiple-choice questions using Google Gemini or Groq AI
- **🔍 Contextual Accuracy**: Questions are verified against source material for factual accuracy 
- **📊 CSV Export**: Generates question files in a standard format ready to import into learning platforms
- **🌐 Multiple Interfaces**:
  - Web application with progress tracking
  - Telegram bot for mobile access
- **⚙️ Customization**: Adjust the number of questions per page
- **🔄 Parallel Processing**: Efficiently processes large documents using multi-threading

---

## 🏗 Architecture

The project consists of three main components:

```
bot/
├── app.py               # Telegram bot implementation using Groq AI
├── bot_gemini.py        # Telegram bot implementation using Gemini AI
├── main.py              # Flask web application
├── requirements.txt     # Python dependencies
└── templates/           # Web interface HTML templates
    ├── base.html        # Base template with layout and styling
    ├── index.html       # Homepage with upload form
    └── job_status.html  # Job processing status page
```

### Core Classes

| Class | Description |
|-------|-------------|
| `PDFProcessor` | Extracts and processes text from PDF documents |
| `MCQGenerator` | Generates MCQs using AI models (Gemini or Groq) |
| `CSVExporter` | Formats and exports MCQs to CSV format |
| `TelegramBot` | Handles Telegram interactions and commands |
| `JobStatus` | Manages processing state for web interface |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Pip package manager
- API keys for Google Gemini or Groq AI
- Telegram Bot token (optional, for Telegram bot)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/pdf-to-mcq-generator.git
cd pdf-to-mcq-generator
```

### Step 2: Install dependencies

```bash
pip install -r bot/requirements.txt
```

### Step 3: Set up environment variables

Create a `.env` file in the project root with the following variables:

```
# Required for both interfaces
GEMINI_API_KEY=your_gemini_api_key
# OR
GROQ_API_KEY=your_groq_api_key

# Required only for Telegram bot
TELEGRAM_TOKEN=your_telegram_bot_token

# Required only for web interface
SECRET_KEY=your_secure_random_string
```

---

## ⚙️ Configuration

The system can be configured through environment variables and constants at the top of the main script files:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_QUESTIONS_PER_PAGE` | 3 | Minimum questions to generate per page |
| `MAX_QUESTIONS_PER_PAGE` | 10 | Maximum questions to generate per page |
| `GEMINI_MODEL` | "gemini-2.0-flash" | Google Gemini model to use |
| `MAX_WORKERS` | 3 | Maximum number of parallel processing threads |
| `PDF_STORAGE_DIR` | "stored_pdfs" | Directory for PDF storage |
| `CSV_STORAGE_DIR` | "stored_csvs" | Directory for generated CSV files |
| `LOG_DIR` | "logs" | Directory for log files |

---

## 📖 Usage

### Web Interface

1. **Start the web server**:
   ```bash
   cd bot
   python main.py
   ```

2. **Access the web interface** at `http://localhost:5000`

3. **Upload a PDF**:
   - Select a PDF file (max 16MB)
   - Set minimum and maximum questions per page
   - Click "Generate Questions"

4. **Monitor progress** on the status page:
   ![Progress tracking](https://via.placeholder.com/800x150/f8f9fa/0d6efd?text=Progress+Tracking)

5. **Download the CSV** once processing is complete

### Telegram Bot

1. **Start the Telegram bot**:
   ```bash
   # For Gemini AI version
   cd bot
   python bot_gemini.py
   
   # For Groq AI version
   cd bot
   python app.py
   ```

2. **Interact with the bot**:
   - Start a chat with your bot on Telegram
   - Send `/start` to get started
   - Send `/help` to see available commands

3. **Send a PDF file** to the bot

4. **Wait for processing** (the bot will show progress updates)

5. **Receive the CSV file** with generated questions

#### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and see welcome message |
| `/help` | Show help information and usage instructions |
| `/set_questions [min] [max]` | Set the minimum and maximum questions per page |
| `/status` | Check the status of your current processing job |

---

## 🔧 Technical Details

### PDF Processing

The system uses a two-step approach for PDF text extraction:
1. Primary extraction with PyMuPDF (fitz)
2. Fallback to PyPDF2 if PyMuPDF fails

Long documents are automatically chunked to fit within AI model context limits using LangChain's `RecursiveCharacterTextSplitter`.

### MCQ Generation

Questions are generated using either:
- Google's Gemini API (via `google.genai`)
- Groq's API (via the Groq Python client)

The generation process includes:
1. Contextual analysis of the PDF text
2. Creating relevant questions with 4 options each
3. Identifying the correct answer
4. Providing an explanation for the correct answer
5. Verifying accuracy through a secondary AI check

### CSV Output Format

Generated CSVs follow the Practice Test Bulk Question Upload format with the following fields:

```
Question, Question Type, Answer Option 1, Explanation 1, Answer Option 2, 
Explanation 2, Answer Option 3, Explanation 3, Answer Option 4, Explanation 4, 
Answer Option 5, Explanation 5, Answer Option 6, Explanation 6, 
Correct Answers, Overall Explanation, Domain
```

### Parallel Processing

Large documents are processed efficiently using `ThreadPoolExecutor` for concurrent page analysis, with `MAX_WORKERS` controlling the degree of parallelism.

---

## 🔒 Security Considerations

- **File Security**: Uploaded files are stored securely with unique identifiers
- **API Key Protection**: API keys are stored as environment variables, not hardcoded
- **Input Validation**: File types and sizes are validated before processing
- **Error Handling**: Robust error handling prevents system crashes
- **Rate Limiting**: Processing is limited to prevent API abuse
- **File Cleanup**: Temporary files are automatically removed after processing

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>
    <b>Made with ❤️ for educators and learners</b>
  </p>
  <p>
    <i>Need support or have questions? Open an issue on GitHub or contact the maintainers.</i>
  </p>
</div>