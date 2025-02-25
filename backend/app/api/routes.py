from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import csv
import io
from services.content_extractor import content_extractor
from core.crew_tasks import QuestionGenerationCrew

router = APIRouter()
crew = QuestionGenerationCrew()

@router.post("/generate")
async def generate_questions(
    urls: List[str] = Form([]),
    files: List[UploadFile] = File([])
):
    all_questions = []
    
    # Process URLs
    for url in urls:
        if url.strip():
            content = await content_extractor.extract_from_url(url)
            questions = await crew.generate_questions_from_content(content)
            all_questions.extend(questions)
    
    # Process PDFs
    for file in files:
        if file.filename.endswith('.pdf'):
            content = await file.read()
            text = await content_extractor.extract_from_pdf(content)
            questions = await crew.generate_questions_from_content(text)
            all_questions.extend(questions)
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Explanation'])
    
    for q in all_questions:
        writer.writerow([
            q['question'],
            q['options']['A'],
            q['options']['B'],
            q['options']['C'],
            q['options']['D'],
            q['correct_answer'],
            q['explanation']
        ])
    
    return {
        "questions": all_questions,
        "csv": output.getvalue()
    }
