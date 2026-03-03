# LiveKit Agent Deployment Guide

This document serves as a reference for deploying and managing your LiveKit Voice Agent and the Node.js outbound caller API.

---

## 🚀 1. Deploying the Python Agent to LiveKit Cloud

Whenever you make changes to `agent.py`, update prompts, or modify `services.json`, you must deploy the new code to LiveKit Cloud so that the production workers pick up the changes.

### Prerequisites

1. Ensure the **LiveKit CLI** is installed on your system. If not, install it using:
   ```bash
   winget install LiveKit.LiveKitCLI
   ```
2. Authenticate with your LiveKit Cloud account:
   ```bash
   lk cloud auth
   ```

### Deployment Steps

1. Open your terminal and navigate to the agent directory:
   ```bash
   cd livekit-voice-agent
   ```
2. Ensure your `.env.local` file is present and contains the required secrets:
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `GOOGLE_API_KEY`
   - `Trunk_ID`
   - Any other necessary API keys.
3. Run the deployment command:
   ```bash
   lk agent deploy
   ```
4. The CLI will package your code, build the Docker container using the existing `Dockerfile`, and deploy it to your `antimatter` project. 
5. Wait for the `Build completed` and `Deployed agent` confirmation messages.

*(Note: You no longer need to run `uv run agent.py start` locally unless you are doing local debugging, as the code runs entirely on LiveKit's servers!)*

---

## 📞 2. Running the Node.js API Server

The Node.js server (`outbound-caller`) is responsible for triggering outbound calls using the LiveKit SDK. It instructs the LiveKit server to dial a specific phone number and assign the `dummyCode` agent to that room.

### Starting the Server

1. Open a new terminal window and navigate to the caller directory:
   ```bash
   cd outbound-caller
   ```
2. Start the Express server:
   ```bash
   node index.js
   ```
3. The server will start running on `http://localhost:3000`. Keep this terminal open.

### Triggering an Outbound Call

To initiate a call, send a `POST` request to the `/call` endpoint with the target phone number in the JSON body. 

**Using curl from PowerShell:**
```powershell
curl -X POST http://localhost:3000/call `
  -H "Content-Type: application/json" `
  -d '{"phone_number": "+917986923834"}'
```

**Using Node.js fetch (JavaScript):**
```javascript
fetch("http://localhost:3000/call", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ phone_number: "+917986923834" })
})
.then(res => res.json())
.then(data => console.log(data));
```

The server will auto-generate a unique room name (e.g., `outbound-12345678`), create the dispatch, and the agent on LiveKit Cloud will immediately execute it and place the SIP call to the phone number.
