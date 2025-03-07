import os
import logging
import tempfile
from dotenv import load_dotenv
import csv
from typing import List, Dict, Any, Optional, Tuple
import io
import json
import re
import datetime
import pathlib
import traceback
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Flask
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
from werkzeug.utils import secure_filename

# PDF processing
import PyPDF2
import fitz  # PyMuPDF

# Google Gemini API
from google import genai
from google.genai import types

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get API tokens from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-me")

# Constants
MIN_QUESTIONS_PER_PAGE = 3
MAX_QUESTIONS_PER_PAGE = 10
GEMINI_MODEL = "gemini-2.0-flash"
PDF_STORAGE_DIR = "stored_pdfs"
LOG_DIR = "logs"
CSV_STORAGE_DIR = "stored_csvs"
MAX_WORKERS = 3  # Maximum number of concurrent page processing
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}

def setup_dirs():
    """Create necessary directories if they don't exist"""
    for directory in [PDF_STORAGE_DIR, LOG_DIR, CSV_STORAGE_DIR, UPLOAD_FOLDER]:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# Create necessary directories before setting up logging
setup_dirs()

# Setup file logging
file_handler = logging.FileHandler(f"{LOG_DIR}/app_{datetime.datetime.now().strftime('%Y%m%d')}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


class PDFProcessor:
    """Class to handle PDF processing operations"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Dict[int, str]:
        """
        Extract text from PDF file, organized by page
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and text content as values
        """
        text_by_page = {}
        
        try:
            # Try using PyMuPDF (fitz) first, which generally has better text extraction
            doc = fitz.open(file_path)
            
            # Get total pages for better logging
            total_pages = len(doc)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Clean up text - remove excessive whitespace
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    
                    text_by_page[page_num + 1] = page_text
                    
                    # Log progress for large documents
                    if (page_num + 1) % 10 == 0 or page_num + 1 == total_pages:
                        logger.info(f"Extracted {page_num + 1}/{total_pages} pages")
                        
                except Exception as e:
                    logger.error(f"Error extracting page {page_num + 1}: {e}")
                    text_by_page[page_num + 1] = ""  # Add empty string for failed pages
                    
            doc.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed, falling back to PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    for page_num in range(total_pages):
                        try:
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            
                            # Clean up text
                            if page_text:
                                page_text = re.sub(r'\s+', ' ', page_text).strip()
                            else:
                                page_text = ""
                                
                            text_by_page[page_num + 1] = page_text
                            
                        except Exception as page_error:
                            logger.error(f"PyPDF2 error on page {page_num + 1}: {page_error}")
                            text_by_page[page_num + 1] = ""  # Add empty string for failed pages
            except Exception as pdf_error:
                logger.error(f"Complete failure in PDF extraction: {pdf_error}")
                raise
        
        # Remove empty pages
        text_by_page = {k: v for k, v in text_by_page.items() if v and len(v.strip()) > 50}
        
        return text_by_page
    
    @staticmethod
    def chunk_text_if_needed(text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Split text into chunks if it exceeds the maximum size
        
        Args:
            text: The text to potentially split
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Use LangChain's text splitter for more intelligent splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return text_splitter.split_text(text)


class MCQGenerator:
    """Class for generating MCQ questions using Google Gemini API"""
    
    def __init__(self, gemini_api_key: str, model_name: str = GEMINI_MODEL):
        """
        Initialize the MCQ generator
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: Gemini model name to use
        """
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.gemini_api_key)

        
    def generate_mcqs_for_text(self, text: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate MCQ questions for the given text
        
        Args:
            text: Source text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of dictionaries containing MCQ questions and answers
        """
        try:
            # Prepare the prompt with explicit JSON format instructions
            prompt = f"""
            Generate exactly {num_questions} multiple-choice questions based on the following text. 
            
            Text: {text}
            
            For each question:
            1. Create a clear, concise question
            2. Provide exactly 4 answer options (labeled A, B, C, D)
            3. Mark the correct answer with its letter (A, B, C, or D)
            4. Write a brief explanation for why the correct answer is right
            
            Format each MCQ in this JSON structure:
            [
                {{
                    "question": "The question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Why the correct answer is right"
                }},
                ...
            ]
            
            IMPORTANT: Make sure each correct_answer is only A, B, C, or D corresponding to the options order.
            Return ONLY the JSON array with no additional text.
            """
            
            # Create the contents for the model
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure the generation parameters - lower temperature for more predictable outputs
            generate_content_config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4000,
                response_mime_type="application/json",  # Request JSON format explicitly
            )
            
            # Call the Gemini model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Extract the response text
            result = response.text
            
            # Parse the result to extract structured MCQ data
            mcq_data = self._parse_llm_response(result)
            
            # Validate and filter the MCQ data
            mcq_data = self._validate_mcqs(mcq_data)
            
            # Add verification step after generating questions
            verified_mcqs = []
            for mcq in mcq_data:
                is_valid, verified_mcq = self.verify_mcq(mcq, text)
                if is_valid:
                    verified_mcqs.append(verified_mcq)
            
            return verified_mcqs[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating MCQs: {e}")
            logger.error(traceback.format_exc())
            return []

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """
        Parse the LLM response to extract structured MCQ data
        
        Args:
            response: Raw text response from the LLM
            
        Returns:
            List of dictionaries with structured MCQ data
        """
        # Try different parsing strategies in sequence
        try:
            # First attempt: direct JSON parsing
            try:
                # Remove any markdown code block markers if present
                clean_response = re.sub(r'```json|```', '', response)
                clean_response = clean_response.strip()
                return json.loads(clean_response)
            except json.JSONDecodeError:
                # Second attempt: Find JSON array in the response
                json_match = re.search(r'\[.*\]', response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            
            # Third attempt: Find individual JSON objects
            json_objects = []
            pattern = r'\{[^{}]*\}'
            matches = re.finditer(pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    obj = json.loads(match.group(0))
                    if 'question' in obj and 'options' in obj:
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
                    
            if json_objects:
                return json_objects
            
            # Fourth attempt: Manual parsing
            questions = []
            current_question = {}
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                
                # Look for question pattern
                if re.match(r'^(\d+\.|Q\d+:|\*\*\d+\*\*\.)', line):
                    if current_question and 'question' in current_question:
                        questions.append(current_question)
                    current_question = {'options': []}
                    current_question['question'] = re.sub(r'^(\d+\.|Q\d+:|\*\*\d+\*\*\.)\s*', '', line)
                
                # Look for option pattern
                option_match = re.match(r'^([A-D])[\.\)]\s*(.+)$', line)
                if option_match and 'options' in current_question:
                    option_letter, option_text = option_match.groups()
                    current_question['options'].append(option_text)
                    
                # Look for correct answer pattern
                correct_match = re.match(r'^Correct Answer:?\s*([A-D])', line)
                if correct_match:
                    current_question['correct_answer'] = correct_match.group(1)
                
                # Look for explanation
                expl_match = re.match(r'^Explanation:?\s*(.+)$', line)
                if expl_match:
                    current_question['explanation'] = expl_match.group(1)
            
            # Add the last question if it exists
            if current_question and 'question' in current_question:
                questions.append(current_question)
                
            return questions
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {response[:200]}...")
            return []
    
    def _validate_mcqs(self, mcqs: List[Dict]) -> List[Dict]:
        """
        Validate MCQs and fix common issues
        
        Args:
            mcqs: List of MCQ dictionaries
            
        Returns:
            List of validated MCQ dictionaries
        """
        valid_mcqs = []
        
        for mcq in mcqs:
            try:
                # Check required fields are present
                if not all(key in mcq for key in ['question', 'options', 'correct_answer']):
                    continue
                
                # Ensure options is a list with 4 items
                if not isinstance(mcq['options'], list) or len(mcq['options']) != 4:
                    continue
                
                # Ensure correct_answer is a single letter A-D
                if not isinstance(mcq['correct_answer'], str) or mcq['correct_answer'] not in 'ABCD':
                    continue
                
                # Ensure explanation is present (add a generic one if missing)
                if 'explanation' not in mcq or not mcq['explanation']:
                    mcq['explanation'] = f"The correct answer is {mcq['correct_answer']}."
                    
                # Strip whitespace from all text fields
                mcq['question'] = mcq['question'].strip()
                mcq['options'] = [opt.strip() for opt in mcq['options']]
                mcq['explanation'] = mcq['explanation'].strip()
                
                valid_mcqs.append(mcq)
                
            except Exception as e:
                logger.error(f"Error validating MCQ: {e}")
                continue
                
        return valid_mcqs
    
    def generate_mcqs_from_pages(self, pages_text: Dict[int, str], min_questions: int, max_questions: int) -> Dict[int, List[Dict]]:
        """
        Generate MCQs for each page in the PDF using parallel processing
        
        Args:
            pages_text: Dictionary with page numbers as keys and text content as values
            min_questions: Minimum number of questions per page
            max_questions: Maximum number of questions per page
            
        Returns:
            Dictionary with page numbers as keys and lists of MCQ dictionaries as values
        """
        result = {}
        futures = []
        
        # Process each page in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for page_num, text in pages_text.items():
                # Skip empty pages
                if not text or len(text.strip()) < 100:
                    continue
                
                # Determine number of questions based on text length
                text_length = len(text)
                if text_length < 500:
                    num_questions = min_questions
                elif text_length > 2000:
                    num_questions = max_questions
                else:
                    # Scale questions based on text length
                    scale_factor = (text_length - 500) / 1500
                    num_questions = min_questions + int(scale_factor * (max_questions - min_questions))
                
                # Submit the task to the executor
                future = executor.submit(
                    self._process_page, 
                    page_num=page_num, 
                    text=text, 
                    num_questions=num_questions
                )
                futures.append(future)
            
            # Process completed futures
            for future in as_completed(futures):
                try:
                    page_num, mcqs = future.result()
                    if mcqs:
                        result[page_num] = mcqs
                except Exception as e:
                    logger.error(f"Error in page processing: {e}")
        
        return result

    def _process_page(self, page_num: int, text: str, num_questions: int) -> Tuple[int, List[Dict]]:
        """
        Process a single page to generate MCQs
        
        Args:
            page_num: Page number
            text: Page text content
            num_questions: Number of questions to generate
            
        Returns:
            Tuple of (page_num, mcqs)
        """
        logger.info(f"Processing page {page_num} - generating {num_questions} questions")
        
        try:
            # For longer texts, split into chunks and process each
            processor = PDFProcessor()
            chunks = processor.chunk_text_if_needed(text)
            
            if len(chunks) == 1:
                # Simple case - just generate MCQs from the single chunk
                mcqs = self.generate_mcqs_for_text(text, num_questions)
            else:
                # Complex case - distribute questions among chunks
                questions_per_chunk = max(1, num_questions // len(chunks))
                mcqs = []
                
                for i, chunk in enumerate(chunks):
                    # For the last chunk, get any remaining questions
                    if i == len(chunks) - 1:
                        remaining = num_questions - len(mcqs)
                        if remaining > 0:
                            chunk_mcqs = self.generate_mcqs_for_text(chunk, remaining)
                            mcqs.extend(chunk_mcqs)
                    else:
                        chunk_mcqs = self.generate_mcqs_for_text(chunk, questions_per_chunk)
                        mcqs.extend(chunk_mcqs)
                    
                    # Stop if we have enough questions
                    if len(mcqs) >= num_questions:
                        break
            
            logger.info(f"Page {page_num} - generated {len(mcqs)} questions")
            return page_num, mcqs
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            logger.error(traceback.format_exc())
            return page_num, []

    def verify_mcq(self, mcq: Dict, source_text: str) -> Tuple[bool, Dict]:
        """
        Verify MCQ accuracy using Gemini API
        """
        try:
            verify_prompt = f"""
            Verify this multiple-choice question for accuracy. Source text and question details below.
            
            Source Text: {source_text}
            
            Question: {mcq['question']}
            Options:
            A. {mcq['options'][0]}
            B. {mcq['options'][1]}
            C. {mcq['options'][2]}
            D. {mcq['options'][3]}
            Given Answer: {mcq['correct_answer']}
            Given Explanation: {mcq['explanation']}
            
            Please verify:
            1. Is the question clear and relevant to the source text?
            2. Is the correct answer accurate?
            3. Is the explanation valid?
            4. Are all options plausible?
            
            Return JSON format:
            {{
                "is_valid": true/false,
                "corrected_answer": "A/B/C/D",
                "corrected_explanation": "explanation",
                "confidence": 0-100
            }}
            """
            
            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=verify_prompt)]
            )]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            verification = json.loads(response.text)
            
            if verification['is_valid'] and verification['confidence'] >= 90:
                if verification['corrected_answer'] != mcq['correct_answer']:
                    mcq['correct_answer'] = verification['corrected_answer']
                    mcq['explanation'] = verification['corrected_explanation']
                return True, mcq
            return False, mcq
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, mcq


class CSVExporter:
    """Class for exporting MCQs to CSV format"""
    
    @staticmethod
    def determine_domain(question: str, text_content: str) -> str:
        """
        Determine the domain/category for a question based on its content
        
        Args:
            question: The question text
            text_content: The source text content
            
        Returns:
            Domain category string
        """
        # List of common educational domains/subjects
        domains = {
            "Mathematics": ["equation", "calculate", "solve", "number", "math", "formula", "computation"],
            "Science": ["experiment", "scientific", "theory", "research", "laboratory", "observation"],
            "Technology": ["computer", "software", "hardware", "digital", "internet", "programming", "technology"],
            "History": ["historical", "century", "period", "war", "civilization", "ancient", "modern"],
            "Literature": ["author", "book", "novel", "character", "story", "literature", "writing"],
            "Business": ["management", "finance", "marketing", "business", "company", "organization"],
            "Engineering": ["design", "system", "process", "engineering", "mechanical", "electrical"],
            "Healthcare": ["medical", "health", "patient", "treatment", "diagnosis", "clinical"],
            "General Knowledge": ["general", "basic", "fundamental", "concept", "definition"]
        }
        
        # Convert text to lowercase for matching
        question_lower = question.lower()
        text_lower = text_content.lower()
        
        # Try to match domain based on keywords
        for domain, keywords in domains.items():
            if any(keyword in question_lower or keyword in text_lower for keyword in keywords):
                return domain
        
        # Default to "Subject Matter" if no specific domain is matched
        return "Subject Matter"
    
    @staticmethod
    def format_mcqs_for_csv(mcqs_by_page: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Format MCQs into the required CSV structure
        """
        csv_rows = []
        letter_to_number = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6"}
        
        # Combine all text content for better domain detection
        all_text = ""
        for mcqs in mcqs_by_page.values():
            for mcq in mcqs:
                all_text += f" {mcq.get('question', '')} {' '.join(mcq.get('options', []))}"
        
        # Initialize Gemini client for domain verification
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        
        for page_num, mcqs in mcqs_by_page.items():
            for mcq in mcqs:
                try:
                    # Get the correct answer letter and convert to number
                    correct_letter = mcq.get("correct_answer", "")
                    correct_number = letter_to_number.get(correct_letter, "1")
                    
                    # Determine domain for this question
                    domain = CSVExporter.determine_domain(mcq.get("question", ""), all_text)
                    
                    # Verify domain classification
                    verified_domain = CSVExporter.verify_domain_classification(
                        mcq.get("question", ""),
                        domain,
                        genai_client
                    )
                    
                    # Create row with determined domain
                    row = {
                        "Question": mcq.get("question", ""),
                        "Question Type": "multiple-choice",
                        "Correct Answers": correct_number,
                        "Overall Explanation": mcq.get("explanation", ""),
                        "Domain": verified_domain  # Use verified domain instead of page number
                    }
                    
                    # Initialize all option fields, including empty ones
                    for i in range(6):
                        option_key = f"Answer Option {i+1}"
                        explanation_key = f"Explanation {i+1}"
                        
                        # Set default empty values
                        row[option_key] = ""
                        row[explanation_key] = ""
                    
                    # Fill in actual values
                    options = mcq.get("options", [])
                    for i, option in enumerate(options):
                        if i < 6:
                            row[f"Answer Option {i+1}"] = option
                            if i == ord(correct_letter) - ord('A'):
                                row[f"Explanation {i+1}"] = mcq.get("explanation", "")
                    
                    csv_rows.append(row)
                except Exception as e:
                    logger.error(f"Error formatting MCQ for CSV: {e}")
                    logger.error(traceback.format_exc())
        
        return csv_rows
    
    @staticmethod
    def create_csv_string(csv_data: List[Dict]) -> str:
        """
        Create a CSV string from MCQ data
        
        Args:
            csv_data: List of dictionaries with MCQ data formatted for CSV
            
        Returns:
            CSV as string
        """
        # Prepare CSV columns based on the Practice Test Bulk Question Upload Template
        fieldnames = [
            "Question", "Question Type", 
            "Answer Option 1", "Explanation 1",
            "Answer Option 2", "Explanation 2",
            "Answer Option 3", "Explanation 3",
            "Answer Option 4", "Explanation 4",
            "Answer Option 5", "Explanation 5",
            "Answer Option 6", "Explanation 6",
            "Correct Answers", "Overall Explanation", "Domain"
        ]
        
        # Create CSV in memory
        output = io.StringIO()
        
        # Sanitize data and ensure all fields exist
        sanitized_data = []
        for row in csv_data:
            sanitized_row = {field: "" for field in fieldnames}  # Initialize with empty values
            for key, value in row.items():
                if key in fieldnames:  # Only include fields in our fieldnames list
                    if value is None:
                        sanitized_row[key] = ""
                    else:
                        sanitized_row[key] = str(value)
            sanitized_data.append(sanitized_row)
        
        # Create the CSV writer and write data
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in sanitized_data:
            writer.writerow(row)
        
        return output.getvalue()

    @staticmethod
    def verify_domain_classification(question: str, domain: str, genai_client) -> str:
        """
        Verify domain classification using Gemini API
        """
        try:
            verify_prompt = f"""
            Verify the domain classification for this question:
            Question: {question}
            Assigned Domain: {domain}
            
            Available domains:
            ALL IT Realted Sub Domains
            
            Return only the most accurate domain name from the list above.
            """
            
            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=verify_prompt)]
            )]
            
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            verified_domain = response.text.strip()
            if verified_domain in [
                "Mathematics", "Science", "Technology", "History",
                "Literature", "Business", "Engineering", "Healthcare",
                "General Knowledge"
            ]:
                return verified_domain
                
        except Exception as e:
            logger.error(f"Domain verification error: {e}")
            
        return domain

# Dictionary to store processing jobs
active_jobs = {}

# Job status class
class JobStatus:
    def __init__(self, file_id, original_filename):
        self.file_id = file_id
        self.original_filename = original_filename
        self.status = "Queued"
        self.message = "Job queued and waiting to start"
        self.progress = 0
        self.csv_path = None
        self.total_questions = 0
        self.pages_processed = 0
        self.start_time = time.time()
        self.complete_time = None
        self.error = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf_file(file_path, job_id, min_questions, max_questions):
    """Process a PDF file and generate MCQs"""
    job = active_jobs[job_id]
    
    try:
        # Update job status
        job.status = "Processing"
        job.message = "Extracting text from PDF"
        job.progress = 10
        
        # Initialize PDF processor and MCQ generator
        pdf_processor = PDFProcessor()
        mcq_generator = MCQGenerator(gemini_api_key=GEMINI_API_KEY)
        csv_exporter = CSVExporter()
        
        # Extract text from PDF
        pages_text = pdf_processor.extract_text_from_pdf(file_path)
        
        # Update job status
        if not pages_text:
            job.status = "Failed"
            job.message = "Could not extract any text from the PDF"
            job.error = "No text extracted"
            return
        
        job.pages_processed = len(pages_text)
        job.message = f"Extracted text from {len(pages_text)} pages, generating questions"
        job.progress = 30
        
        # Generate MCQs from pages
        mcqs_by_page = mcq_generator.generate_mcqs_from_pages(
            pages_text,
            min_questions,
            max_questions
        )
        
        # Check if generation was successful
        total_questions = sum(len(mcqs) for mcqs in mcqs_by_page.values())
        
        if total_questions == 0:
            job.status = "Failed"
            job.message = "Could not generate any questions from the PDF"
            job.error = "No questions generated"
            return
        
        # Update job status
        job.total_questions = total_questions
        job.message = f"Generated {total_questions} questions, creating CSV"
        job.progress = 80
        
        # Format and export as CSV
        csv_data = csv_exporter.format_mcqs_for_csv(mcqs_by_page)
        csv_string = csv_exporter.create_csv_string(csv_data)
        
        # Create CSV file
        original_name = os.path.splitext(job.original_filename)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}_{original_name}_questions.csv"
        csv_filepath = os.path.join(CSV_STORAGE_DIR, csv_filename)
        
        # Write CSV with UTF-8-BOM for Excel compatibility
        with open(csv_filepath, 'w', encoding='utf-8-sig') as f:
            f.write(csv_string)
        
        # Update job status as complete
        job.status = "Complete"
        job.message = "Processing complete"
        job.progress = 100
        job.csv_path = csv_filepath
        job.complete_time = time.time()
        
        logger.info(f"Job {job_id} completed: {total_questions} questions generated")
        
    except Exception as e:
        logger.error(f"Error processing PDF (job {job_id}): {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update job status with error
        job.status = "Failed"
        job.message = f"Error during processing: {str(e)}"
        job.error = str(e)


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create app routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'pdf_file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['pdf_file']
    
    # Check if user submitted an empty form
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        flash('Only PDF files are allowed', 'error')
        return redirect(request.url)
    
    try:
        # Get parameters
        min_questions = int(request.form.get('min_questions', MIN_QUESTIONS_PER_PAGE))
        max_questions = int(request.form.get('max_questions', MAX_QUESTIONS_PER_PAGE))
        
        # Validate parameters
        if min_questions < 1 or min_questions > 10:
            min_questions = MIN_QUESTIONS_PER_PAGE
        if max_questions < min_questions or max_questions > 20:
            max_questions = MAX_QUESTIONS_PER_PAGE
        
        # Create a unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)
        
        # Create job status
        job = JobStatus(job_id, filename)
        active_jobs[job_id] = job
        
        # Start processing thread
        thread = threading.Thread(
            target=process_pdf_file,
            args=(file_path, job_id, min_questions, max_questions)
        )
        thread.daemon = True
        thread.start()
        
        flash('File uploaded and processing started!', 'success')
        return redirect(url_for('job_status', job_id=job_id))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        logger.error(f"Upload error: {str(e)}")
        return redirect(request.url)

@app.route('/job/<job_id>')
def job_status(job_id):
    # Check if job exists
    if job_id not in active_jobs:
        flash('Job not found', 'error')
        return redirect(url_for('index'))
    
    job = active_jobs[job_id]
    return render_template('job_status.html', job=job, job_id=job_id)

@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    # Return job status as JSON for AJAX updates
    if job_id not in active_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = active_jobs[job_id]
    
    return jsonify({
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "total_questions": job.total_questions,
        "pages_processed": job.pages_processed,
        "original_filename": job.original_filename,
        "complete": job.status in ["Complete", "Failed"],
        "error": job.error
    })

@app.route('/download/<job_id>')
def download_file(job_id):
    # Check if job exists and is complete
    if job_id not in active_jobs:
        flash('Job not found', 'error')
        return redirect(url_for('index'))
    
    job = active_jobs[job_id]
    
    if job.status != "Complete" or not job.csv_path:
        flash('CSV not yet available or job failed', 'error')
        return redirect(url_for('job_status', job_id=job_id))
    
    # Get the original filename without extension
    original_name = os.path.splitext(job.original_filename)[0]
    download_name = f"{original_name}_questions.csv"
    
    # Send the file
    return send_file(
        job.csv_path,
        mimetype='text/csv',
        download_name=download_name,
        as_attachment=True
    )

@app.route('/cleanup/<job_id>')
def cleanup_job(job_id):
    # Remove job and files after download
    if job_id in active_jobs:
        job = active_jobs[job_id]
        
        # Try to remove the upload file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{job.original_filename}")
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception as e:
                logger.error(f"Error removing upload file: {str(e)}")
        
        # If completed successfully, keep the CSV for 1 hour
        if job.status == "Complete" and time.time() - job.complete_time > 3600:
            if job.csv_path and os.path.exists(job.csv_path):
                try:
                    os.remove(job.csv_path)
                except Exception as e:
                    logger.error(f"Error removing CSV file: {str(e)}")
            
            # Remove the job
            del active_jobs[job_id]
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)