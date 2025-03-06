import os
from google import genai
from google.genai import types

def generate():
    """
    Simple standalone script to test Gemini API functionality
    """
    # Initialize the Gemini API with the key
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCxnQ4dLk-KuACERZFlgr4r3yCflMmNgHs")
    client = genai.Client(api_key=api_key)
    
    # Specify the model
    model = "gemini-2.0-flash"
    
    # Create a prompt for MCQ generation
    sample_text = """
    The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. 
    Water can change states among liquid, vapor, and ice at various places in the water cycle. Although the balance of water on Earth remains fairly constant over time, 
    individual water molecules can come and go. The water moves from one reservoir to another, such as from river to ocean, 
    or from the ocean to the atmosphere, by the physical processes of evaporation, condensation, precipitation, infiltration, surface runoff, and subsurface flow. 
    In doing so, the water goes through different forms: liquid, solid (ice) and vapor.
    """
    
    prompt = f"""
    Generate 3 multiple-choice questions based on the following text.
    
    Text: {sample_text}
    
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
    
    # Create the content for the model
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
    
    # Call the model and print each chunk of the response
    print("Generating MCQs with Gemini model...\n")
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
    print("\n\nGeneration complete!")

if __name__ == "__main__":
    generate()