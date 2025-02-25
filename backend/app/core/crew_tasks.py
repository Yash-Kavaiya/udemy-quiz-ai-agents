from crewai import Task, Agent
from core.gemini import gemini_service

class QuestionGenerationCrew:
    def __init__(self):
        self.researcher = Agent(
            role='Researcher',
            goal='Extract relevant content from URLs and PDFs',
            backstory='Expert at analyzing and extracting key information from various sources',
            allow_delegation=False
        )
        
        self.question_creator = Agent(
            role='Question Creator',
            goal='Create high-quality multiple choice questions',
            backstory='Expert at creating engaging and educational questions',
            allow_delegation=False
        )

    async def generate_questions_from_content(self, content: str) -> list:
        research_task = Task(
            description=(
                f"Analyze this content and extract key information: {content}\n"
                "Focus on important concepts, facts, and relationships."
            ),
            agent=self.researcher
        )

        question_task = Task(
            description=(
                "Create multiple choice questions based on the analyzed content."
                "Each question should have 4 options and an explanation."
            ),
            agent=self.question_creator
        )

        # Use Gemini for actual question generation
        questions = await gemini_service.generate_questions(content)
        return questions
