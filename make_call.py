import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]

ngrok_url = os.environ["NGROK_URL"]
my_phone_number = os.environ["MY_PHONE_NUMBER"]
twilio_phone_number = os.environ["TWILIO_PHONE_NUMBER"]

client = Client(account_sid, auth_token)

call = client.calls.create(
    record=True,

    url=f"https://{ngrok_url}/twiml",
    to=my_phone_number,
    from_=twilio_phone_number,

    recording_status_callback=f"https://{ngrok_url}/recording-complete",
    recording_status_callback_method="POST",
)

print(call.sid)