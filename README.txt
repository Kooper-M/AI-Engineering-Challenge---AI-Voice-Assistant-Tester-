To initialize, first fill out the .env file with the necessary API keys and credentials according to the .env.example file.

SETUP: 3 terminals are required.
1. In one terminal, start the ngrok server with:
   ngrok http --url=YOUR_NGROK_URL 8080
2. In the second terminal, run:
   py app.py
   to start the Uvicorn server with the code instructions.
3. In the third terminal, run:
   py make_call.py
   to start the patient call.
