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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Telegram libraries
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Bot, InputFile, Update

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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
MIN_QUESTIONS_PER_PAGE = 3
MAX_QUESTIONS_PER_PAGE = 10
GEMINI_MODEL = "gemini-2.0-flash"
PDF_STORAGE_DIR = "stored_pdfs"
LOG_DIR = "logs"
CSV_STORAGE_DIR = "stored_csvs"
MAX_WORKERS = 3  # Maximum number of concurrent page processing

def setup_dirs():
    """Create necessary directories if they don't exist"""
    for directory in [PDF_STORAGE_DIR, LOG_DIR, CSV_STORAGE_DIR]:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# Create necessary directories before setting up logging
setup_dirs()

# Setup file logging
file_handler = logging.FileHandler(f"{LOG_DIR}/bot_{datetime.datetime.now().strftime('%Y%m%d')}.log")
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
            
            # Limit to requested number
            return mcq_data[:num_questions]
            
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


class CSVExporter:
    """Class for exporting MCQs to CSV format"""
    
    @staticmethod
    def format_mcqs_for_csv(mcqs_by_page: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Format MCQs into the required CSV structure for the Practice Test Bulk Question Upload Template
        
        Args:
            mcqs_by_page: Dictionary with page numbers as keys and lists of MCQ dictionaries as values
            
        Returns:
            List of dictionaries formatted for CSV export
        """
        csv_rows = []
        
        for page_num, mcqs in mcqs_by_page.items():
            for mcq in mcqs:
                try:
                    # Create a row with all required fields
                    row = {
                        "Question": mcq.get("question", ""),
                        "Question Type": "Multiple Choice",
                        "Correct Answers": mcq.get("correct_answer", ""),
                        "Overall Explanation": mcq.get("explanation", ""),
                        "Domain": f"Page {page_num}"
                    }
                    
                    # Add options and explanations
                    options = mcq.get("options", [])
                    option_letters = ["A", "B", "C", "D", "E", "F"]
                    correct_letter = mcq.get("correct_answer", "")
                    
                    # Initialize all option fields, including empty ones
                    for i in range(6):
                        option_key = f"Answer Option {i+1}"
                        explanation_key = f"Explanation {i+1}"
                        
                        # Set default empty values
                        row[option_key] = ""
                        row[explanation_key] = ""
                    
                    # Fill in actual values
                    for i, option in enumerate(options):
                        if i < len(option_letters):
                            letter = option_letters[i]
                            row[f"Answer Option {i+1}"] = option
                            
                            # Add explanation for correct answer only
                            if letter == correct_letter:
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
    def format_questions_text(mcqs_by_page: Dict[int, List[Dict]]) -> str:
        """
        Format MCQs into a readable text format
        
        Args:
            mcqs_by_page: Dictionary with page numbers as keys and lists of MCQ dictionaries as values
            
        Returns:
            Formatted string with all questions
        """
        text_output = "📝 GENERATED QUESTIONS\n\n"
        
        for page_num, mcqs in sorted(mcqs_by_page.items()):
            text_output += f"📄 Page {page_num}\n\n"
            
            for i, mcq in enumerate(mcqs, 1):
                text_output += f"Question {i}: {mcq.get('question', '')}\n"
                
                # Add options
                options = mcq.get("options", [])
                option_letters = ["A", "B", "C", "D"]
                
                for j, option in enumerate(options):
                    if j < len(option_letters):
                        letter = option_letters[j]
                        text_output += f"   {letter}. {option}\n"
                
                # Add correct answer
                correct = mcq.get("correct_answer", "")
                text_output += f"Correct Answer: {correct}\n"
                
                # Add explanation
                explanation = mcq.get("explanation", "")
                if explanation:
                    text_output += f"Explanation: {explanation}\n"
                
                text_output += "\n"
            
            text_output += "\n"
        
        return text_output


class TelegramBot:
    """Class for the Telegram bot implementation"""
    
    def __init__(self, telegram_token: str, mcq_generator: MCQGenerator):
        """
        Initialize the Telegram bot
        
        Args:
            telegram_token: Telegram Bot API token
            mcq_generator: MCQ generator instance
        """
        self.token = telegram_token
        self.mcq_generator = mcq_generator
        self.pdf_processor = PDFProcessor()
        self.csv_exporter = CSVExporter()
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command
        """
        user = update.effective_user
        logger.info(f"User {user.id} ({user.username}) started the bot")
        
        await update.message.reply_html(
            f"👋 <b>Hello, {user.first_name}!</b>\n\n"
            f"I can generate multiple-choice questions from PDF files using Google's Gemini model.\n\n"
            f"Just send me a PDF document, and I'll analyze it and create quiz questions in CSV format "
            f"compatible with the Practice Test Bulk Upload Template.\n\n"
            f"The process may take a few minutes depending on the size of your PDF."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /help command
        """
        await update.message.reply_text(
            "📚 <b>Bot Usage Instructions</b>\n\n"
            "1. <b>Send a PDF file</b> - I'll automatically start processing it\n"
            "2. <b>Wait for processing</b> - This may take several minutes for larger PDFs\n"
            "3. <b>Receive your CSV file</b> - Ready to be uploaded to learning platforms\n\n"
            "<b>Available Commands:</b>\n"
            "• /start - Start the bot\n"
            "• /help - Show this help message\n"
            "• /set_questions [min] [max] - Customize the number of questions per page\n"
            "  Example: /set_questions 3 10\n"
            "• /status - Check the status of your current processing job\n\n"
            "<b>CSV Format:</b> The output will match the Practice Test Bulk Question Upload Template",
            parse_mode="HTML"
        )
    
    async def set_questions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /set_questions command to customize min/max questions
        """
        try:
            # Get command arguments
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "⚠️ Please specify both minimum and maximum questions per page.\n"
                    "Example: /set_questions 3 10"
                )
                return
            
            min_q = int(args[0])
            max_q = int(args[1])
            
            # Validate input
            if min_q < 1 or min_q > 10 or max_q < min_q or max_q > 20:
                await update.message.reply_text(
                    "⚠️ Invalid values. Min must be at least 1, max must be at least min and no more than 20."
                )
                return
            
            # Store in user data
            if not context.user_data:
                context.user_data = {}
            
            context.user_data["min_questions"] = min_q
            context.user_data["max_questions"] = max_q
            
            await update.message.reply_text(
                f"✅ Settings updated! I'll generate between {min_q} and {max_q} questions per page."
            )
            
        except ValueError:
            await update.message.reply_text("⚠️ Please provide valid numbers for min and max questions.")
        except Exception as e:
            logger.error(f"Error in set_questions: {e}")
            await update.message.reply_text("❌ An error occurred while updating settings.")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /status command to check current processing status
        """
        # Check if there's a processing job in the user data
        processing_status = context.user_data.get("processing_status", None) if context.user_data else None
        
        if not processing_status:
            await update.message.reply_text(
                "No active processing job. Send me a PDF to start generating questions."
            )
        else:
            await update.message.reply_text(
                f"Current status: {processing_status}"
            )
    
    async def process_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Process uploaded PDF files
        """
        # Check if the message contains a document
        if not update.message.document:
            await update.message.reply_text("Please send a PDF file.")
            return
        
        # Check if the file is a PDF
        document = update.message.document
        if not document.file_name.lower().endswith('.pdf'):
            await update.message.reply_text("Please send a PDF file. I can only process PDF documents.")
            return
        
        # Get user info for logging
        user = update.effective_user
        logger.info(f"User {user.id} ({user.username}) uploaded file: {document.file_name}")
        
        # Check if file size is too large (Telegram limit is 20MB)
        if document.file_size > 20 * 1024 * 1024:
            await update.message.reply_text(
                "⚠️ This PDF is larger than 20MB, which is the maximum size Telegram allows.\n"
                "Please try a smaller file."
            )
            return
        
        # Send a processing message with more detail
        processing_msg = await update.message.reply_text(
            f"🔄 Processing your PDF: <b>{document.file_name}</b>\n\n"
            f"• Step 1/4: Downloading file...\n"
            f"• Step 2/4: Pending - Extract text from PDF\n"
            f"• Step 3/4: Pending - Generate MCQs using Gemini AI\n"
            f"• Step 4/4: Pending - Create CSV output\n\n"
            f"This may take several minutes for larger documents.",
            parse_mode="HTML"
        )
        
        # Initialize processing status in user data
        if not context.user_data:
            context.user_data = {}
        context.user_data["processing_status"] = "Downloading PDF"
        
        try:
            # Download the file
            file = await context.bot.get_file(document.file_id)
            
            # Create a safe filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_\.]', '_', document.file_name)
            stored_filename = f"{timestamp}_{safe_filename}"
            stored_path = os.path.join(PDF_STORAGE_DIR, stored_filename)
            
            # Save the file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                pdf_path = temp_file.name
                await file.download_to_drive(pdf_path)
                
                # Copy to permanent storage
                with open(pdf_path, 'rb') as src, open(stored_path, 'wb') as dst:
                    dst.write(src.read())
                    
            logger.info(f"PDF saved: {stored_path}")
            
            # Get min/max questions from context or use defaults
            min_questions = context.user_data.get("min_questions", MIN_QUESTIONS_PER_PAGE) if context.user_data else MIN_QUESTIONS_PER_PAGE
            max_questions = context.user_data.get("max_questions", MAX_QUESTIONS_PER_PAGE) if context.user_data else MAX_QUESTIONS_PER_PAGE
            
            # Update status for text extraction
            context.user_data["processing_status"] = "Extracting text from PDF"
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"🔄 Processing your PDF: <b>{document.file_name}</b>\n\n"
                     f"• Step 1/4: ✅ File downloaded successfully\n"
                     f"• Step 2/4: 🔄 Extracting text from PDF...\n"
                     f"• Step 3/4: Pending - Generate MCQs using Gemini AI\n"
                     f"• Step 4/4: Pending - Create CSV output",
                parse_mode="HTML"
            )
            
            # Extract text from PDF
            pages_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Check if we successfully extracted text
            if not pages_text:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id, 
                    message_id=processing_msg.message_id,
                    text=f"❌ Could not extract any text from this PDF.\n\n"
                         f"This could be because:\n"
                         f"• The PDF contains only images or scanned content\n"
                         f"• The PDF is password-protected\n"
                         f"• The PDF format is not standard\n\n"
                         f"Please try a different PDF file with proper text content."
                )
                # Clean up temporary file
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                return
            
            # Update status for MCQ generation
            context.user_data["processing_status"] = "Generating MCQs"
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"🔄 Processing your PDF: <b>{document.file_name}</b>\n\n"
                     f"• Step 1/4: ✅ File downloaded successfully\n"
                     f"• Step 2/4: ✅ Extracted text from {len(pages_text)} pages\n"
                     f"• Step 3/4: 🔄 Generating questions using Gemini AI...\n"
                     f"• Step 4/4: Pending - Create CSV output\n\n"
                     f"This step may take a few minutes. The bot is generating between "
                     f"{min_questions} and {max_questions} questions per page.",
                parse_mode="HTML"
            )
            
            # Generate MCQs
            mcqs_by_page = self.mcq_generator.generate_mcqs_from_pages(
                pages_text, 
                min_questions, 
                max_questions
            )
            
            # Check if generation was successful
            total_questions = sum(len(mcqs) for mcqs in mcqs_by_page.values())
            
            if total_questions == 0:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id, 
                    message_id=processing_msg.message_id,
                    text=f"❌ Could not generate any questions from this PDF.\n\n"
                         f"This could be because:\n"
                         f"• The text content is too short or lacks sufficient information\n"
                         f"• The content may not be suitable for multiple-choice questions\n"
                         f"• There may have been an API error with Gemini\n\n"
                         f"Please try a different PDF with more substantial content."
                )
                # Clean up temporary file
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                return
            
            # Update status for CSV creation
            context.user_data["processing_status"] = "Creating CSV output"
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"🔄 Processing your PDF: <b>{document.file_name}</b>\n\n"
                     f"• Step 1/4: ✅ File downloaded successfully\n"
                     f"• Step 2/4: ✅ Extracted text from {len(pages_text)} pages\n"
                     f"• Step 3/4: ✅ Generated {total_questions} questions\n"
                     f"• Step 4/4: 🔄 Creating CSV output...",
                parse_mode="HTML"
            )
            
            # Format and export CSV
            csv_data = self.csv_exporter.format_mcqs_for_csv(mcqs_by_page)
            csv_string = self.csv_exporter.create_csv_string(csv_data)
            
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
            
            # Prepare CSV filename based on original PDF name
            original_name = os.path.splitext(document.file_name)[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{timestamp}_{original_name}_questions.csv"
            csv_filepath = os.path.join(CSV_STORAGE_DIR, csv_filename)
            
            # Write CSV with UTF-8-BOM for Excel compatibility
            with open(csv_filepath, 'w', encoding='utf-8-sig') as f:
                f.write(csv_string)
            
            # Mark processing as complete
            context.user_data["processing_status"] = "Complete"
            
            # Update the message to show completion
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"✅ Processing complete for: <b>{document.file_name}</b>\n\n"
                     f"• Step 1/4: ✅ File downloaded successfully\n"
                     f"• Step 2/4: ✅ Extracted text from {len(pages_text)} pages\n"
                     f"• Step 3/4: ✅ Generated {total_questions} questions\n"
                     f"• Step 4/4: ✅ CSV created successfully\n\n"
                     f"Sending your CSV file now...",
                parse_mode="HTML"
            )
            
            # Send CSV file to user
            with open(csv_filepath, 'rb') as f:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=f,
                    filename=csv_filename,
                    caption=f"📊 Here's your CSV file with {total_questions} questions in the Practice Test Bulk Upload format."
                )
            
            # Send completion message with statistics
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"✅ <b>Processing Summary</b>\n\n"
                     f"• PDF: {document.file_name}\n"
                     f"• Pages processed: {len(pages_text)}\n"
                     f"• Questions generated: {total_questions}\n"
                     f"• Average questions per page: {total_questions / len(pages_text):.1f}\n\n"
                     f"The CSV is ready for upload to your learning platform. Send another PDF when you're ready!",
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            
            # Try to provide a helpful error message based on the exception
            error_message = str(e)
            if "Token limit exceeded" in error_message:
                user_message = "The PDF content was too large for processing. Please try a smaller PDF or one with less content."
            elif "API key" in error_message.lower():
                user_message = "There was an issue with the AI service credentials. Please contact the bot administrator."
            elif "timeout" in error_message.lower():
                user_message = "The request timed out. This might be due to a large PDF or high server load. Please try again later."
            else:
                user_message = f"An error occurred while processing your PDF. Please try again later or with a different file."
            
            # Update the message to show the error
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"❌ <b>Error Processing PDF</b>\n\n{user_message}",
                parse_mode="HTML"
            )
            
            # Clean up any temporary files
            try:
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            except Exception:
                pass
    
    def run(self) -> None:
        """
        Run the Telegram bot
        """
        # Create application
        application = Application.builder().token(self.token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("set_questions", self.set_questions))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(MessageHandler(filters.Document.PDF, self.process_pdf))
        
        # Add a general message handler for invalid inputs
        application.add_handler(MessageHandler(
            ~filters.Command() & ~filters.Document.PDF, 
            lambda update, context: update.message.reply_text(
                "I only understand PDF files and commands. Please send me a PDF file or use /help to see available commands."
            )
        ))
        
        # Run the bot
        logger.info("Starting bot with polling...")
        application.run_polling(allowed_updates=["message"])


def main() -> None:
    """
    Main function to start the bot
    """
    # Create necessary directories
    setup_dirs()
    
    # Check environment variables
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN must be set in .env file")
        print("Error: TELEGRAM_TOKEN not set in .env file. Please add your Telegram bot token.")
        return
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY must be set in .env file")
        print("Error: GEMINI_API_KEY not set in .env file. Please add your Google Gemini API key.")
        return
    
    # Initialize MCQ generator with Gemini
    mcq_generator = MCQGenerator(gemini_api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    
    # Initialize and run Telegram bot
    bot = TelegramBot(telegram_token=TELEGRAM_TOKEN, mcq_generator=mcq_generator)
    
    logger.info("Starting bot...")
    print(f"Bot started. Press Ctrl+C to exit.")
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        print(f"Error running bot: {e}")


if __name__ == "__main__":
    main()