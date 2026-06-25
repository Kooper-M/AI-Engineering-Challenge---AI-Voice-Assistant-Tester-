import os
import json
from pickle import GET
import uvicorn
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", "8080"))
DOMAIN = os.getenv("NGROK_URL")
WS_URL = f"wss://{DOMAIN}/ws"
MODEL = "gpt-4o-mini" # You can change this to any OpenAI model you prefer
WELCOME_GREETING = "Hi, I was hoping to schedule an appointment."
BASE_SYSTEM_PROMPT = """
You are pretending to be a real patient calling a medical office phone agent.

Speak naturally and briefly. This conversation is being converted to voice, so every response should sound good when spoken aloud.

Do not mention that you are an AI, a bot, Twilio, OpenAI, code, prompts, or testing.

Do not use emojis, bullet points, asterisks, markdown, or special formatting.

Keep most responses to one or two short sentences.

If you do not understand the agent, ask a natural clarification like, "Sorry, could you say that again?"

Stay focused on the current patient goal and gently steer the conversation back if needed.
"""

SCENARIO_PROMPT = """
Current scenario: scheduling an appointment.

You are Alex Johnson. You are calling because your knee has been hurting for a few months.

Your goal is to schedule a non-emergency appointment.

Preferences:
You prefer Friday morning.
If that is not available, you can accept Monday afternoon, Tuesday morning, or Wednesday after two.
Do not accept weekends unless the agent says the office is open weekends.
If the agent offers something unreasonable, ask a natural follow-up.

Patient details:
Date of birth: January fifth, two thousand one.
Phone number: nine one two, three nine eight, one four seven five.
Insurance: Blue Cross Blue Shield.
Symptoms: knee pain, clicking sometimes, worse going down stairs.

Conversation style:
Be cooperative, but not robotic.
Answer only what the agent asks.
If the agent needs more detail, give it naturally.
When the appointment is confirmed, thank them and end the call.
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + SCENARIO_PROMPT


RECORDING_DIR = Path("recordings")
RECORDING_DIR.mkdir(exist_ok=True)

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

def save_transcript_line(call_sid: str, speaker: str, text: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = TRANSCRIPT_DIR / f"{call_sid}.txt"

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
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return completion.choices[0].message.content

@app.post("/twiml")
async def twiml_endpoint():
    """Endpoint that returns TwiML for Twilio to connect to the WebSocket"""
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Connect>
        <ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" ttsProvider="ElevenLabs" voice="FGY2WhTYpPnrIDTdsKH5" />
      </Connect>
    </Response>"""
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    call_sid = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                websocket.call_sid = call_sid
                sessions[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                
            elif message["type"] == "prompt":
                agent_text = message["voicePrompt"]
                print(f"Processing prompt: {agent_text}")

                conversation = sessions[websocket.call_sid]

                # The other side of the call said this
                conversation.append({"role": "user", "content": agent_text})

                # Your bot generates a patient-style response
                response = await ai_response(conversation)

                # Your bot's response is still the OpenAI "assistant"
                conversation.append({"role": "assistant", "content": response})

                # Transcript labels can use real-world names
                save_transcript_line(websocket.call_sid, "Agent", agent_text)
                save_transcript_line(websocket.call_sid, "Patient Bot", response)

                await websocket.send_text(
                    json.dumps({
                        "type": "text",
                        "token": response,
                        "last": True
                    })
                )

                print(f"Sent response: {response}")
                
            elif message["type"] == "interrupt":
                print("Handling interruption.")
                
            else:
                print(f"Unknown message type received: {message['type']}")
                
    except WebSocketDisconnect:
        print("WebSocket connection closed")
        if call_sid:
            sessions.pop(call_sid, None)

@app.post("/recording-complete")
async def recording_complete(request: Request):
    """Twilio calls this after the recording is finished processing."""
    form = await request.form()

    call_sid = form.get("CallSid")
    recording_sid = form.get("RecordingSid")
    recording_url = form.get("RecordingUrl")

    print("Recording complete")
    print("Call SID:", call_sid)
    print("Recording SID:", recording_sid)
    print("Recording URL:", recording_url)

    recording_file_path = RECORDING_DIR / f"{call_sid}.mp3"
    mp3_url = f"{recording_url}.mp3"

    response = requests.get(
        mp3_url,
        auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    )

    response.raise_for_status()

    with open(recording_file_path, "wb") as file:
        file.write(response.content)

    print(f"Saved recording to {recording_file_path}")

    return Response(status_code=204)

if __name__ == "__main__":
    print(f"Server running at http://localhost:{PORT} and {WS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)