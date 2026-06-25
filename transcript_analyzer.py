from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

def analyze_transcript(call_sid: str):
    transcript_file = Path("transcripts") / f"{call_sid}.txt"
    
    if not transcript_file.exists():
        print(f"Transcript file for call SID {call_sid} does not exist.")
        return
    

    """Analyze the transcript using OpenAI API and save the analysis to a file."""
    with open(transcript_file, "r", encoding="utf-8") as file:
        transcript_text = file.read()

    prompt = """You are a QA analyst reviewing a transcript of a phone call between a simulated patient and a medical office phone agent.

Your task is to analyze the transcript and identify any bugs, failures, or quality issues in the conversation.

Focus on:

* Whether the patient completed the intended scenario goal
* Whether the agent misunderstood the patient
* Whether the agent gave incorrect, unsafe, or unrealistic information
* Whether the agent failed to follow normal medical office behavior
* Whether there were turn-taking issues, awkward pauses, interruptions, or transcription problems
* Whether the patient bot sounded unnatural or failed to stay in character

Return your analysis in this format:

Call Summary:
Briefly summarize what happened in the call.

Scenario Outcome:
Say whether the patient successfully completed the goal. Explain why or why not.

Issues Found:
List any bugs or quality issues. For each issue, include:
Bug:
Severity: Low, Medium, or High
Where:
Details:
Expected Behavior:

Overall Sentiment:
Briefly describe the tone of the interaction.

Final Recommendation:
Say whether this call should be considered successful, partially successful, or failed.

If there are no major bugs, say that clearly. Do not invent bugs that are not supported by the transcript."""

    completion = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=[ 
            {
                "role": "system", "content": "You are a QA analyst reviewing voice AI phone call transcripts." },
            {
                "role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}"},
        ],
    )
    analysis = completion.choices[0].message.content
    output_path = ANALYSIS_DIR / f"{call_sid}_analysis.txt"
    output_path.write_text(analysis, encoding="utf-8")

    print(f"Analysis for call SID {call_sid} saved to {output_path}")

if __name__ == "__main__":
    analyze_transcript("test_call_sid")  #replaced when needed

