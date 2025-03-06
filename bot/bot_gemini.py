import os
import logging
import tempfile
from dotenv import load_dotenv
import csv
from typing import List, Dict, Any
import io
import json
import re

# Telegram libraries
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Bot, InputFile

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCxnQ4dLk-KuACERZFlgr4r3yCflMmNgHs")

# Constants
MIN_QUESTIONS_PER_PAGE = 3
MAX_QUESTIONS_PER_PAGE = 10
GEMINI_MODEL = "gemini-2.0-flash"
PDF_STORAGE_DIR = "stored_pdfs"
LOG_DIR = "logs"

import datetime
import pathlib

def setup_dirs():
    """Create necessary directories if they don't exist"""
    for directory in [PDF_STORAGE_DIR, LOG_DIR]:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# Create necessary directories before setting up logging
setup_dirs()

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
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_by_page[page_num + 1] = page.get_text()
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed, falling back to PyPDF2: {e}")
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_by_page[page_num + 1] = page.extract_text()
        
        return text_by_page


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
            # Prepare the prompt
            prompt = f"""
            Generate {num_questions} multiple-choice questions based on the following text. 
            
            Text: {text}
            
            For each question:
            1. Create a clear, concise question
            2. Provide exactly 4 answer options (labeled A, B, C, D)
            3. Mark the correct answer
            4. Write a brief explanation for why the correct answer is right
            
            Format each MCQ in this structure (as a JSON array):
            [
                {{
                    "question": "The question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Why the correct answer is right"
                }},
                ...
            ]
            
            Return only the JSON array with no additional text.
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
            
            # Configure the generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4000,
                response_mime_type="text/plain",
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
            
            return mcq_data
            
        except Exception as e:
            logger.error(f"Error generating MCQs: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """
        Parse the LLM response to extract structured MCQ data
        
        Args:
            response: Raw text response from the LLM
            
        Returns:
            List of dictionaries with structured MCQ data
        """
        import re
        import json
        
        # Try to extract JSON objects from the response
        try:
            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # If no JSON array found, try to find individual JSON objects
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
            
            # If still no JSON, try manual parsing as a last resort
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
            return []
    
    def generate_mcqs_from_pages(self, pages_text: Dict[int, str], min_questions: int, max_questions: int) -> Dict[int, List[Dict]]:
        """
        Generate MCQs for each page in the PDF
        
        Args:
            pages_text: Dictionary with page numbers as keys and text content as values
            min_questions: Minimum number of questions per page
            max_questions: Maximum number of questions per page
            
        Returns:
            Dictionary with page numbers as keys and lists of MCQ dictionaries as values
        """
        result = {}
        
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
            
            # Generate MCQs for this page
            mcqs = self.generate_mcqs_for_text(text, num_questions)
            
            # Store in results
            if mcqs:
                result[page_num] = mcqs
        
        return result


class CSVExporter:
    """Class for exporting MCQs to CSV format"""
    
    @staticmethod
    def format_mcqs_for_csv(mcqs_by_page: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Format MCQs into the required CSV structure
        
        Args:
            mcqs_by_page: Dictionary with page numbers as keys and lists of MCQ dictionaries as values
            
        Returns:
            List of dictionaries formatted for CSV export
        """
        csv_rows = []
        
        for page_num, mcqs in mcqs_by_page.items():
            for mcq in mcqs:
                try:
                    # Create a new row entry with all required fields
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
                    
                    for i, option in enumerate(options):
                        if i < len(option_letters):
                            letter = option_letters[i]
                            row[f"Answer Option {i+1}"] = option
                            # If this is the correct answer, add the explanation
                            if mcq.get("correct_answer") == letter:
                                row[f"Explanation {i+1}"] = mcq.get("explanation", "")
                            else:
                                row[f"Explanation {i+1}"] = ""
                    
                    csv_rows.append(row)
                except Exception as e:
                    logger.error(f"Error formatting MCQ for CSV: {e}")
        
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
        # Prepare CSV columns based on the template
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
        
        # Sanitize data
        sanitized_data = []
        for row in csv_data:
            sanitized_row = {}
            for key, value in row.items():
                if value is None:
                    sanitized_row[key] = ""
                else:
                    sanitized_row[key] = str(value)
            sanitized_data.append(sanitized_row)
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in sanitized_data:
            # Ensure all fields are present
            csv_row = {field: "" for field in fieldnames}
            csv_row.update(row)
            writer.writerow(csv_row)
        
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
    
    async def start(self, update, context) -> None:
        """
        Handle the /start command
        """
        await update.message.reply_html(
            f"Hi! I can generate multiple-choice questions from PDF files using Google's Gemini model.\n\n"
            "Just send me a PDF document, and I'll analyze it and create quiz questions in CSV format.\n\n"
            "The process may take a few minutes depending on the size of your PDF."
        )
    
    async def help_command(self, update, context) -> None:
        """
        Handle the /help command
        """
        await update.message.reply_text(
            "To use this bot:\n\n"
            "1. Send a PDF file\n"
            "2. Wait while I process it (this may take a few minutes)\n"
            "3. I'll send you a CSV file with multiple-choice questions\n\n"
            "You can customize the number of questions per page by using command: /set_questions min max\n"
            "Example: /set_questions 3 10"
        )
    
    async def set_questions(self, update, context) -> None:
        """
        Handle the /set_questions command to customize min/max questions
        """
        try:
            # Get command arguments
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "Please specify both minimum and maximum questions per page.\n"
                    "Example: /set_questions 3 10"
                )
                return
            
            min_q = int(args[0])
            max_q = int(args[1])
            
            # Validate input
            if min_q < 1 or min_q > 10 or max_q < min_q or max_q > 20:
                await update.message.reply_text(
                    "Invalid values. Min must be at least 1, max must be at least min and no more than 20."
                )
                return
            
            # Store in user data
            if not context.user_data:
                context.user_data = {}
            
            context.user_data["min_questions"] = min_q
            context.user_data["max_questions"] = max_q
            
            await update.message.reply_text(
                f"Settings updated. I'll generate between {min_q} and {max_q} questions per page."
            )
            
        except ValueError:
            await update.message.reply_text("Please provide valid numbers for min and max questions.")
        except Exception as e:
            logger.error(f"Error in set_questions: {e}")
            await update.message.reply_text("An error occurred while updating settings.")
    
    async def process_pdf(self, update, context) -> None:
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
        
        # Send a processing message
        processing_msg = await update.message.reply_text("Processing your PDF. This may take a few minutes...")
        
        try:
            # Download the file
            file = await context.bot.get_file(document.file_id)
            
            # Save PDF to storage directory with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_\.]', '_', document.file_name)
            stored_filename = f"{timestamp}_{safe_filename}"
            stored_path = os.path.join(PDF_STORAGE_DIR, stored_filename)
            
            # Save temporary and permanent copies
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                pdf_path = temp_file.name
                await file.download_to_drive(pdf_path)
                # Copy to storage
                with open(pdf_path, 'rb') as src, open(stored_path, 'wb') as dst:
                    dst.write(src.read())
            
            logger.info(f"PDF saved: {stored_path}")
            
            # Log user information
            user = update.effective_user
            logger.info(f"Processing PDF for user: {user.id} ({user.username})")
            
            # Get min/max questions from context or use defaults
            min_questions = context.user_data.get("min_questions", MIN_QUESTIONS_PER_PAGE) if context.user_data else MIN_QUESTIONS_PER_PAGE
            max_questions = context.user_data.get("max_questions", MAX_QUESTIONS_PER_PAGE) if context.user_data else MAX_QUESTIONS_PER_PAGE
            
            # Update status
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text="PDF downloaded. Extracting text..."
            )
            
            # Extract text from PDF
            pages_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Update status
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"Text extracted from {len(pages_text)} pages. Generating questions using Google Gemini..."
            )
            
            # Generate MCQs
            mcqs_by_page = self.mcq_generator.generate_mcqs_from_pages(
                pages_text, 
                min_questions, 
                max_questions
            )
            
            # Update status
            total_questions = sum(len(mcqs) for mcqs in mcqs_by_page.values())
            if total_questions == 0:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id, 
                    message_id=processing_msg.message_id,
                    text="Could not generate any questions from this PDF. It may contain too little text or be in a format I can't process."
                )
                return
            
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"Generated {total_questions} questions. Preparing output..."
            )
            
            # Format and export
            csv_data = self.csv_exporter.format_mcqs_for_csv(mcqs_by_page)
            csv_string = self.csv_exporter.create_csv_string(csv_data)
            questions_text = self.csv_exporter.format_questions_text(mcqs_by_page)
            
            # Delete temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
            
            # Send text file with questions first
            original_name = os.path.splitext(document.file_name)[0]
            
            # Use open() to create temporary files
            questions_filename = f"{original_name}_questions.txt"
            temp_text_path = os.path.join(tempfile.gettempdir(), questions_filename)
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                f.write(questions_text)
            
            with open(temp_text_path, 'rb') as f:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=f,
                    filename=questions_filename,
                    caption=f"Here's a text file with all {total_questions} questions."
                )
            
            # Clean up temporary file
            if os.path.exists(temp_text_path):
                os.unlink(temp_text_path)
            
            # Send CSV file
            csv_filename = f"{original_name}_questions.csv"
            temp_csv_path = os.path.join(tempfile.gettempdir(), csv_filename)
            
            # Write CSV with BOM for Excel compatibility
            with open(temp_csv_path, 'w', encoding='utf-8-sig') as f:
                f.write(csv_string)
            
            with open(temp_csv_path, 'rb') as f:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=f,
                    filename=csv_filename,
                    caption=f"Here's the CSV file with {total_questions} questions in the Udemy quiz format."
                )
            
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
            
            # Send completion message
            await context.bot.delete_message(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id
            )
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"✅ Processing complete! Generated {total_questions} questions from {len(pages_text)} pages using Google Gemini."
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=processing_msg.message_id,
                text=f"An error occurred while processing your PDF: {str(e)}"
            )
    
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
        application.add_handler(MessageHandler(filters.Document.PDF, self.process_pdf))
        
        # Run the bot
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
        return
    
    # Initialize MCQ generator with Gemini
    mcq_generator = MCQGenerator(gemini_api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    
    # Initialize and run Telegram bot
    bot = TelegramBot(telegram_token=TELEGRAM_TOKEN, mcq_generator=mcq_generator)
    
    logger.info("Starting bot...")
    bot.run()


if __name__ == "__main__":
    main()