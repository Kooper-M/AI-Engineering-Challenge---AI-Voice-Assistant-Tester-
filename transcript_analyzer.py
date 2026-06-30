from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CALLS_DIR = Path("calls")
CALLS_DIR.mkdir(exist_ok=True)

def get_call_dir(call_sid: str) -> Path:
    call_dir = CALLS_DIR / call_sid
    call_dir.mkdir(exist_ok=True)
    return call_dir

def analyze_transcript(call_sid: str, scenario_prompt: str):
    call_dir = get_call_dir(call_sid)
    transcript_file = call_dir / "transcript.txt"
    legacy_transcript_file = Path("transcripts") / f"{call_sid}.txt"

    if not transcript_file.exists() and legacy_transcript_file.exists():
        transcript_file = legacy_transcript_file
    
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
* Whether there were turn-taking issues, awkward pauses, or interruptions that changed the outcome
* Whether the patient bot sounded unnatural or failed to stay in character
* Whether the call state stayed consistent across the transcript

Before writing the analysis, infer a brief call-state timeline for yourself:
requested goal, identity details provided, original appointment if any, action taken, final appointment or next step.

Use that timeline to detect contradictions, repeated actions, premature closing, wrong-date offers, missing final confirmation, or cases where the agent offers the same appointment/date/provider that was just canceled or rescheduled away from without clearly explaining why it is a different valid option.

Apply goal-specific checks:

* For cancellation and rescheduling calls, verify the correct appointment was canceled or moved, the replacement appointment is not confusingly the same appointment, and the final date, time, provider or location, and next step were confirmed.
* For scheduling calls, verify the agent collected enough information, offered a valid slot, and confirmed the appointment date, time, provider or location, and next step.
* For refill calls, verify the patient learned whether the refill request will be submitted, when to expect it, and whether anything else is needed.
* For insurance, location, records, hours, or unusual request calls, verify the specific question was answered or a clear next step was given.

Do not treat likely speech-to-text artifacts as bugs by themselves. Minor misspellings, phonetic doctor names, repeated filler words, duplicated recording disclosures, or garbled words should only be mentioned if they materially changed the call outcome, caused a real misunderstanding, or made the agent take the wrong action. Prefer focusing on behavior and task completion over transcript spelling.

Doctor/provider names are especially likely to be transcribed phonetically. Treat variants such as "Zigniew", "Zebigniew", "Zignew", "Zivignu", "Lakoski", "Lukowski", or similar-sounding names as the same provider unless the transcript clearly shows that the wrong provider was booked, the patient objected in a way that changed the outcome, or the agent took a different action because of the name confusion.

Hard exclusion: do not include any issue whose only evidence is inconsistent spelling, pronunciation, or transcription of a provider name. Do not write speculative language like "might create confusion" for provider-name variants. If the patient did not change course because of the name and the final appointment/provider was otherwise clear, omit the issue entirely.

Return your analysis in this format:

Call Summary:
Briefly summarize what happened in the call.

Scenario Outcome:
Include "Goal completed: Yes", "Goal completed: No", or "Goal completed: Partially". Explain why or why not, separately from any quality issues.

Issues Found:
List any bugs or quality issues. Before listing an issue, verify it is not excluded by the speech-to-text or provider-name rules above. For each issue, include:
Bug:
Severity: Low, Medium, or High
Source: Agent, Patient Bot, Transcription, or Unknown
Where:
Details:
Expected Behavior:

Severity guidance:
High means the goal failed, the wrong appointment/action occurred, unsafe or materially incorrect information was given, or verification failed in a way that blocked the call.
Medium means the goal completed but there was meaningful friction, confusion, contradiction, repeated questioning, premature closing, or missing clarity.
Low means polish issues that did not materially affect the outcome.

Overall Sentiment:
Briefly describe the tone of the interaction.

Final Recommendation:
Say whether this call should be considered successful, partially successful, or failed.

Before finalizing, review your Issues Found section and remove any issue based only on provider-name spelling, pronunciation, or transcription variation. If there are no major bugs, say that clearly. Do not invent bugs that are not supported by the transcript."""

    completion = client.chat.completions.create(
        model="gpt-5",
        messages=[ 
            {
                "role": "system", "content": "You are a QA analyst reviewing voice AI phone call transcripts. Follow the exclusion rules exactly; do not report provider-name spelling, pronunciation, or transcription variants as bugs unless they changed the outcome." },
            {
                "role": "user", "content": f"{prompt}\n\nScenario goal:\n{scenario_prompt}\n\nTranscript:\n{transcript_text}"},
        ],
    )
    analysis = completion.choices[0].message.content
    output_path = call_dir / "analysis.txt"
    
    output_text = (
        "SCENARIO PROMPT\n"
        "================\n"
        f"{scenario_prompt}\n\n\n"
        "ANALYSIS\n"
        "================\n"
        f"{analysis}\n"
    )

    output_path.write_text(output_text, encoding="utf-8")

    print(f"Analysis for call SID {call_sid} saved to {output_path}")

if __name__ == "__main__":
    analyze_transcript("CA3f475568d8790ee1fd0f5578e7a5152f", "canceling an appointment. You are calling because you believe you have a checkup that needs to be canceled due to illness, then you want the earliest available weekday slot instead. If the office finds an existing appointment, try to cancel that appointment after confirming it is yours. If the office says you do not have an appointment to cancel, stop pursuing the cancellation and schedule a new appointment.")  #replaced when needed
