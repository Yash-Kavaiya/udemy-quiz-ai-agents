import google.generativeai as genai
from core.config import settings

class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate_questions(self, content: str, num_questions: int = 5) -> list:
        prompt = f"""
        Generate {num_questions} multiple choice questions based on the following content:
        {content}
        
        Format each question as:
        Q: [Question]
        A: [Correct Answer]
        B: [Wrong Answer]
        C: [Wrong Answer]
        D: [Wrong Answer]
        Correct Answer: A/B/C/D
        Explanation: [Brief explanation why the answer is correct]
        
        Return in CSV format with headers:
        Question,Option A,Option B,Option C,Option D,Correct Answer,Explanation
        """
        
        response = await self.model.generate_content_async(prompt)
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> list:
        # Parse the CSV formatted response
        lines = text.strip().split('\n')
        questions = []
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                try:
                    q, a, b, c, d, ans, exp = line.split(',')
                    questions.append({
                        'question': q.strip(),
                        'options': {
                            'A': a.strip(),
                            'B': b.strip(),
                            'C': c.strip(),
                            'D': d.strip()
                        },
                        'correct_answer': ans.strip(),
                        'explanation': exp.strip()
                    })
                except Exception:
                    continue
                    
        return questions

gemini_service = GeminiService()
