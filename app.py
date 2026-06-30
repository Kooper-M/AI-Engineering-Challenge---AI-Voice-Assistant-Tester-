import os
import json
import asyncio
import uvicorn
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import Response
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from transcript_analyzer import analyze_transcript

# Load environment variables from .env file
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", "8080"))
DOMAIN = os.getenv("NGROK_URL")
WS_URL = f"wss://{DOMAIN}/ws"
MODEL = "gpt-4o-mini" # You can change this to any OpenAI model you prefer
PROMPT_PADDING_SECONDS = float(os.getenv("PROMPT_PADDING_SECONDS", "1.5"))
CURRENT_DATE_TEXT = datetime.now().strftime("%B %d, %Y")
BASE_SYSTEM_PROMPT = """
You are pretending to be a real patient calling a medical office phone agent.

Speak naturally and briefly. This conversation is being converted to voice, so every response should sound good when spoken aloud.

Do not mention that you are an AI, a bot, Twilio, OpenAI, code, prompts, or testing.

Do not use emojis, bullet points, asterisks, markdown, or special formatting.

Keep most responses to one or two short sentences.

If you do not understand the agent, ask a natural clarification like, "Sorry, could you say that again?"

Stay focused on the current patient goal and gently steer the conversation back if needed.

You are the patient, not the clinic staff. Be cooperative and realistic. Do not volunteer extra information before the agent asks for it, but do not create unnecessary friction once the agent is helping correctly.

Be cooperative and realistic, but only provide information when asked, ask for clarification when the agent is vague or contradictory, and keep pursuing the patient goal until it is clearly complete.

Do not repeat the full goal in consecutive turns. After you have stated why you are calling, answer the agent's immediate question directly and use short references like "that appointment" or "the reschedule" unless the agent seems confused.

If asked for one piece of information, give only that information. Do not bundle your date of birth, phone number, insurance, symptoms, and scheduling preferences together unless the agent asks for them.

If the agent gives unclear instructions, asks an ambiguous question, gives conflicting dates or times, or tries to end the call before your goal is complete, ask one short natural clarification. Only restate what you still need if a simple clarification would not be enough.

If the agent asks for information you already gave, provide it again once. If they ask again, sound mildly confused but still polite.

If the agent asks you to upload, text, email, fax, or send an insurance card, ID, photo, form, portal document, or other file during the call, say you are not able to do that right now. Ask if there is another way to continue, whether they can note it for later, or what the next step should be.

For cancellation or rescheduling calls, adapt to the appointment record the agent actually finds. If the agent finds an existing appointment on a different date, ask to confirm it is yours and then continue with that appointment instead of insisting on the scripted date. If the agent says there is no matching appointment or no upcoming appointment, stop trying to cancel and ask to schedule the requested new appointment instead.

If asked about emergency symptoms, do not invent severe symptoms. Deny chest pain, severe shortness of breath, fainting, major injury, fever, sudden weakness, or other emergency symptoms unless the current scenario says otherwise.

If the agent gives medical advice instead of helping you schedule, route, or complete your request, ask whether you should be seen by the doctor.
"""

SCENARIO_PROMPT = [
    # 0: scheduling an appointment
    "Current scenario: scheduling an appointment. You are calling because your knee has been hurting for a few months. Your goal is to schedule a non-emergency appointment.",
    # 1: scheduling an appointment
    "Current scenario: scheduling an appointment. You are calling to schedule a follow-up visit for chronic back pain. Your goal is to book a Tuesday morning appointment and explain the pain has lasted 3 months.",
    # 2: scheduling an appointment
    "Current scenario: scheduling an appointment. You are calling to make a new patient appointment for a sore shoulder. Your goal is to find a Monday or Wednesday afternoon slot.",
    # 3: rescheduling an appointment
    "Current scenario: rescheduling an appointment. You are calling because you believe you have a migraine follow-up that needs to be moved because your work schedule changed. You may be misremembering the appointment date. If the office finds an existing appointment, confirm it is yours and try to move that appointment to Tuesday afternoon. If the office says you do not have an appointment to move, stop pursuing the reschedule and ask to schedule a new migraine follow-up for Tuesday afternoon.",
    # 4: canceling an appointment
    "Current scenario: canceling an appointment. You are calling because you believe you have a checkup that needs to be canceled due to illness, then you want the earliest available weekday slot instead. If the office finds an existing appointment, try to cancel that appointment after confirming it is yours. If the office says you do not have an appointment to cancel, stop pursuing the cancellation and schedule a new appointment.",
    # 5: rescheduling an appointment
    "Current scenario: rescheduling an appointment. You are calling to move your appointment because of a family emergency and you need Wednesday or Friday after 2 PM.",
    # 6: medication refill
    "Current scenario: medication refill. You are calling to request a refill for lisinopril. Your goal is to ask if the office can authorize a 30-day refill today.",
    # 7: medication refill
    "Current scenario: medication refill. You are calling to refill your asthma inhaler and ask whether the doctor needs to approve it first because the pharmacy says you are out.",
    # 8: medication refill
    "Current scenario: medication refill. You are calling to request a chronic medication refill and ask whether the doctor can sign off on a 90-day supply.",
    # 9: office hours
    "Current scenario: office hours question. You are calling to ask whether the clinic is open on Saturdays, and if not, what the earliest weekday hours are.",
    # 10: location and insurance
    "Current scenario: locations and insurance. You are calling to get an overview of the doctors and whether that office accepts Blue Cross Blue Shield.",
    # 11: location and insurance
    "Current scenario: locations and insurance. You are a new patient asking if the office accepts your PPO plan, where the nearest location is, and how soon you can be seen for a knee injury.",
    # 12: office hours
    "Current scenario: office hours question. You are calling to ask if the office is open after 5 PM and whether they offer evening appointments; if not, ask for the next available morning slot.",
    # 13: unclear issue
    "Current scenario: unclear issue. You are calling because you feel ‘off’ but are not sure if it is urgent. Your goal is to ask what kind of appointment you should schedule and whether you should come in sooner.",
    # 14: interruption
    "Current scenario: interruption. You are calling while interrupted by a doorbell or another person speaking in the background. Pause briefly, then tell the agent you need one moment but still want to continue the call.",
    # 15: confusion
    "Current scenario: confusion. You are a patient who heard two different dates from the agent and ask for confirmation: 'Just to confirm, is my appointment on June 26 or July 26?'",
    # 16: unusual request
    "Current scenario: unusual request. You are calling to ask if the office can fax records to a specialist or handle a radiology disc.",
    # 17: demographics update
    "Current scenario: updating contact information. You are calling because you think the office may have an old phone number on file. Ask them to update the chart to your actual phone number, plus one, eight seven seven, three four nine, six zero two five. Do not ask the office to read the current phone number on file unless they offer to verify it after confirming your identity. Your goal is to have the office confirm that plus one, eight seven seven, three four nine, six zero two five is saved on your chart or clearly explain the next step to get it updated. If the office needs to verify your identity first, provide only the information they ask for."
]

COMMON_PATIENT_DETAILS = """
Call context:
Today is {current_date}. Use this date when reasoning about appointment dates, relative dates, or whether a date is in the past or future.

Patient details:
Full name: Alex Johnson.
Date of birth: January fifth, two thousand one.
Phone number: plus one, eight seven seven, three four nine, six zero two five.
Insurance: Blue Cross Blue Shield.
"""

APPOINTMENT_PREFERENCES = """
Appointment preferences:
You prefer Friday morning.
If that is not available, you can accept Monday afternoon, Tuesday morning, or Wednesday after two.
Do not accept weekends unless the agent says the office is open weekends.
If the agent offers something unreasonable, ask a natural follow-up.
"""

KNEE_APPOINTMENT_DETAILS = """
Appointment symptoms:
Symptoms: knee pain, clicking sometimes, worse going down stairs.
"""

REFILL_DETAILS = """
Medication and pharmacy details:
For asthma inhaler calls, the medication is albuterol sulfate and you are out of it now.
For lisinopril calls, ask whether the office can authorize a thirty day refill today.
Your usual pharmacy is Walgreens at 2945 South 6th Street, Springfield, Illinois, 62703.
If asked about refill timing, ask when to expect the request to be sent and whether the doctor needs to approve it first.
"""

CONVERSATION_STYLE = """
Conversation style:
Be cooperative, but not robotic.
Answer only what the agent asks.
If the agent needs more detail, give it naturally.
Avoid repeating the same sentence structure or the full appointment request across back-to-back turns.
Occasionally use brief natural hesitation, like "Um," "I think," or "Let me check," but do not be theatrical.
"""

COMPLETION_RULES = """
Completion:
For appointment calls, your goal is complete only when the appointment date, time, location, and next step are clear.
For refill calls, your goal is complete only when you know whether the refill request will be sent, when to expect it, and whether the office needs anything else.
For contact information updates, your goal is complete only when the office confirms the new phone number is saved or clearly explains the next step needed to update it.
For insurance, location, records, hours, or unusual request calls, your goal is complete only when the agent has answered the specific question or clearly explained the next step.
When the goal is complete, thank them and end the call.
"""

SCENARIO_INDEX = int(os.getenv("SCENARIO_INDEX", "0"))
CURRENT_SCENARIO_PROMPT = SCENARIO_PROMPT[SCENARIO_INDEX % len(SCENARIO_PROMPT)]

def scenario_extra_context(scenario_index: int) -> str:
    context = []
    if scenario_index in {0, 1, 2, 3, 4, 5, 11, 12}:
        context.append(APPOINTMENT_PREFERENCES)
    if scenario_index in {0, 11}:
        context.append(KNEE_APPOINTMENT_DETAILS)
    if scenario_index in {6, 7, 8}:
        context.append(REFILL_DETAILS)
    return "\n".join(context)

SYSTEM_PROMPT = "\n".join([
    BASE_SYSTEM_PROMPT,
    COMMON_PATIENT_DETAILS.format(current_date=CURRENT_DATE_TEXT),
    CURRENT_SCENARIO_PROMPT,
    scenario_extra_context(SCENARIO_INDEX % len(SCENARIO_PROMPT)),
    CONVERSATION_STYLE,
    COMPLETION_RULES,
])


CALLS_DIR = Path("calls")
CALLS_DIR.mkdir(exist_ok=True)

def get_call_dir(call_sid: str) -> Path:
    call_dir = CALLS_DIR / call_sid
    call_dir.mkdir(exist_ok=True)
    return call_dir

def save_transcript_line(call_sid: str, speaker: str, text: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = get_call_dir(call_sid) / "transcript.txt"

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {speaker}: {text}\n")

# Initialize OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store active sessions
sessions = {}

# Create FastAPI app
app = FastAPI()

async def ai_response(messages):
    """Get a response from OpenAI API"""
    completion = await asyncio.to_thread(
        openai.chat.completions.create,
        model=MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content

async def process_agent_prompt(websocket: WebSocket, agent_text: str):
    call_sid = websocket.call_sid

    print(f"Processing prompt: {agent_text}")
    #agent response
    conversation = sessions[call_sid]
    conversation.append({"role": "user", "content": agent_text})
    #patient response
    response = await ai_response(conversation)
    conversation.append({"role": "assistant", "content": response})

    # Transcript labels can use real-world names
    save_transcript_line(call_sid, "Agent", agent_text)
    save_transcript_line(call_sid, "Patient Bot", response)

    await websocket.send_text(
        json.dumps({
            "type": "text",
            "token": response,
            "last": True
        })
    )

    print(f"Sent response: {response}")

async def prompt_buffer_worker(websocket: WebSocket, prompt_queue: asyncio.Queue):
    while True:
        agent_texts = [await prompt_queue.get()]

        while True:
            try:
                next_text = await asyncio.wait_for(
                    prompt_queue.get(),
                    timeout=PROMPT_PADDING_SECONDS,
                )
                agent_texts.append(next_text)
            except asyncio.TimeoutError:
                break

        combined_agent_text = " ".join(text.strip() for text in agent_texts if text.strip())
        if combined_agent_text:
            await process_agent_prompt(websocket, combined_agent_text)

@app.post("/twiml")
async def twiml_endpoint():
    """Endpoint that returns TwiML for Twilio to connect to the WebSocket"""
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Connect>
        <ConversationRelay url="{WS_URL}" ttsProvider="ElevenLabs" voice="FGY2WhTYpPnrIDTdsKH5" />
      </Connect>
    </Response>"""
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    call_sid = None
    prompt_queue = asyncio.Queue()
    buffer_task = asyncio.create_task(prompt_buffer_worker(websocket, prompt_queue))
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                get_call_dir(call_sid)
                websocket.call_sid = call_sid
                sessions[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                
            elif message["type"] == "prompt":
                agent_text = message["voicePrompt"]
                print(f"Queued prompt: {agent_text}")
                await prompt_queue.put(agent_text)
                
            elif message["type"] == "interrupt":
                print("Handling interruption.")
                
            else:
                print(f"Unknown message type received: {message['type']}")
                
    except WebSocketDisconnect:
        print("WebSocket connection closed")
        if call_sid:
            sessions.pop(call_sid, None)
    finally:
        buffer_task.cancel()
        try:
            await buffer_task
        except asyncio.CancelledError:
            pass

@app.post("/recording-complete")
async def recording_complete(request: Request, background_tasks: BackgroundTasks):
    """Twilio calls this after the recording is finished processing."""
    form = await request.form()

    call_sid = form.get("CallSid")
    recording_sid = form.get("RecordingSid")
    recording_url = form.get("RecordingUrl")

    print("Recording complete")
    print("Call SID:", call_sid)
    print("Recording SID:", recording_sid)
    print("Recording URL:", recording_url)

    call_dir = get_call_dir(call_sid)
    recording_file_path = call_dir / "recording.mp3"
    mp3_url = f"{recording_url}.mp3"

    response = requests.get(
        mp3_url,
        auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    )

    response.raise_for_status()

    with open(recording_file_path, "wb") as file:
        file.write(response.content)

    print(f"Saved recording to {recording_file_path}")

    background_tasks.add_task(analyze_transcript, call_sid, CURRENT_SCENARIO_PROMPT)

    return Response(status_code=204)

if __name__ == "__main__":
    print(f"Server running at http://localhost:{PORT} and {WS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
