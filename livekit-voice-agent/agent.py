from google.genai.types import TurnCoverage
import os
import json
import logging
import asyncio
import time
from datetime import datetime as dt
from urllib.parse import quote
from typing import Annotated
from dotenv import load_dotenv
from livekit import api
from livekit import rtc
from livekit.agents import (
    Agent,
    function_tool,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    cli,
    room_io,
    WorkerOptions,
)
from livekit.plugins import noise_cancellation
from livekit.plugins import google
from google.genai import types
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import AudioContent

from logger import get_call_logger, log_outbound_call_result

logger = logging.getLogger("agent-dummyCode")

# Mask Livekit's noisy internal Realtime errors that leak past try-except blocks
class SuppressLivekitNudgeTraceback(logging.Filter):
    def __init__(self):
        super().__init__()
        self.active_session = None

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            msg = record.getMessage()
            # Silently suppress realtime reply task errors (normal during user interruptions)
            if "Error in _realtime_reply_task" in msg:
                return False
            # Only trigger recovery for actual socket drops (1008)
            if "error in receive task: 1008" in msg:
                if self.active_session:
                    logger.warning("[ERROR RECOVERY] Gemini 1008 detected. Firing recovery nudge in 2s.")
                    async def global_quick_recovery():
                        await asyncio.sleep(2.0)
                        try:
                            await self.active_session.generate_reply(
                                instructions="The user has been silent for a while. Ask them a short, polite question to check if they are still there and if they need help.",
                                allow_interruptions=True,
                            )
                        except Exception:
                            pass
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(global_quick_recovery())
                    except RuntimeError:
                        pass
                return False
        return True

nudge_filter = SuppressLivekitNudgeTraceback()
logging.getLogger("livekit.agents").addFilter(nudge_filter)
logging.getLogger("livekit.plugins.google").addFilter(nudge_filter)

load_dotenv(".env.local")

# Silence noisy PyMongo heartbeat/topology logs
logging.getLogger("pymongo").setLevel(logging.ERROR)
logging.getLogger("pymongo.topology").setLevel(logging.ERROR)
logging.getLogger("pymongo.connection").setLevel(logging.ERROR)

# Load services once at startup for high-speed access
try:
    with open(os.path.join(os.path.dirname(__file__), "services.json"), 'r', encoding='utf-8') as f:
        SERVICES = json.load(f)
    logger.info(f"Loaded {len(SERVICES)} services into memory.")
except Exception as e:
    logger.error(f"Failed to load services.json: {e}")
    SERVICES = []

# --- S3 RECORDING CONFIG ---
S3_BUCKET = os.getenv("BUCKET", "caller-recordings")
S3_REGION = os.getenv("REGION", "ap-south-1")
S3_ACCESS_KEY = os.getenv("ACCESS_KEY", "")
S3_SECRET = os.getenv("SECRET", "")

def build_s3_url(bucket: str, region: str, key: str) -> str:
    encoded_key = quote(key, safe='/')
    return f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"

async def start_call_recording(ctx: JobContext, call_uuid: str) -> tuple:
    """Start a RoomCompositeEgress recording (audio-only OGG) uploaded to S3."""
    if not S3_ACCESS_KEY or not S3_SECRET:
        logger.warning("S3 credentials not configured. Skipping recording.")
        return "", ""

    try:
        current_date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = f"recording/{call_uuid}/recording_{current_date}.ogg"

        req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[api.EncodedFileOutput(
                file_type=api.EncodedFileType.OGG,
                filepath=filepath,
                s3=api.S3Upload(
                    bucket=S3_BUCKET,
                    region=S3_REGION,
                    access_key=S3_ACCESS_KEY,
                    secret=S3_SECRET
                ),
            )],
        )

        lkapi = api.LiveKitAPI()
        res = await lkapi.egress.start_room_composite_egress(req)
        egress_id = res.egress_id
        logger.info(f"Recording started: egress_id={egress_id}, filepath={filepath}")
        return egress_id, filepath
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        return "", ""
    
class DefaultAgent(Agent):
    def __init__(self, call_logger=None, session_data=None) -> None:
        self.call_logger = call_logger
        self.session_data = session_data

        end_call_tool = EndCallTool(
            extra_description="Only end the call after confirming the customer's issue is resolved or they want to hang up.",
            delete_room=True,
            end_instructions="End with a warm, short goodbye like: 'It was lovely talking to you! Have a wonderful day ahead.' Keep it natural and brief — no more than one sentence.",
        )
        super().__init__(
            tools=end_call_tool.tools,
            instructions="""### System Prompt: Yes Madam Cart Recovery Concierge (Neha)

**ROLE & PERSONA**
You are a voice-based AI Assistant named **Neha**, representing the **Yes Madam** home salon services team. You are a 25-year-old Indian female. Your tone is polite, soothing, helpful, and natural—like a friendly coordinator.
**Your Mission:** Call users who abandoned their app cart, identify their roadblock, ask a quick probing question to understand their needs better, and then pivot to scheduling a priority 1-hour callback from a human expert who can finalize the booking for them.

**GENDER RULES (Strict)**
* **Customer Addressing:** Always be **gender-neutral** when addressing the customer. Do NOT assume their gender. Never use "Ma'am", "Sir", "Madam", "bhaiya", "didi", "bhai" etc. Use neutral terms like "aap", "you", or simply their name if available.
* **Self-Reference:** When referring to yourself (Neha), always use **feminine** language. In Hindi, use feminine verb forms (e.g., "मैं बता रही हूँ", "मैंने सोचा"). In English, use "I" naturally without gender markers.

**LANGUAGE & MIRRORING DYNAMICS**
STRICT ENFORCEMENT RULES:>
* **Accent:** You always use Indian accent and way of talking.
* **Bilingual Approach:** You are fluent in both English and Hindi.
* **The Opening:** You must always start the call with exactly: *"हेलो, नमस्ते?"* and then seamlessly transition into English for the rest of the introduction.
* **Mirroring:** After your introduction, **listen carefully to the customer's language.** If they reply in Hindi or a mix of Hindi/English, switch your entire response to mix of Hindi/English immediately. If they reply in English, continue in English.
* **Strict Script Formatting:** Whenever you speak Hindi, output the text **strictly in Devanagari script** (e.g., "आप कैसे हैं?"). Do not write Hindi using English letters, and **never** output the same phrase twice (i.e., do not output both Devanagari and transliteration).
* **Natural Hindi:** Use simple, everyday conversational Hindi. For example, use "आज का दिन कैसा था". Use the respectful "जी" (ji) only in a few places to sound natural, not in every sentence.
End of rules."

**GEMINI LIVE / REAL-TIME VOICE DIRECTIVES**

* **Conversational & Concise:** Speak in natural, spoken language. Keep all responses under 3 short sentences.
* **Pacing & Flow:** Never stop mid-sentence without a proper ending. Always conclude your thought with a clear question to prompt the user to respond.
* **Handle Interruptions Gracefully:** If the user interrupts, stop immediately, acknowledge what they said, and adapt.
* **No Echoing:** **Do not repeat the customer's statements back to them.** Acknowledge simply with "Right", "Got it", or "I understand", and move forward.

## Important rules
- Always pitch about discounts and offers if customer hesitate to book the call / service.
- Dont tell services pricing until user ask, just keep with yourself and tell when asked.
- Only consider valid inputs to move to the next steps. If anything confusing ask or clarify with the user.
- Always use indian accent when seapking in english or in hindi.
- Talk to the point with the customer. 
- Always never fabricate answers. 
- If you don't know the answer, say so.
- If you don't understand the question, ask for clarification.
- Never froget the context of the conversation.
- If customer ask why you called "tell the complete inforamtion and mention comapny name".
- Always try to understand the user's intent and provide the most relevant information.
- If customer says no, allways try to tell them about the offers and discounts.
- Never tell the price of the services until users ask.
- **CRITICAL TOOL RULE**: If you call a tool and it returns an error or "No services found", you MUST NOT stay silent. You MUST immediately reply to the user apologizing that you couldn't find the information, and ask if they need anything else.
- Below privded script are not mandatory to follow, you can change them as per the conversation.
- Conversation should feel natural and human-like.

** Tool call rules **
- Step 1: ALWAYS use `search_services_summary` first to get a quick visual of available options matching the user's intent. Do not guess prices.
- Step 2: Pitch 1 or 2 options from the summary to the user.
- Step 3: ONLY use `get_service_details` if the user specifically asks "What's in that?", "What are the benefits?", or "How long does it take?" about a specific service.
- Never read out raw JSON or formatting symbols to the user. Speak naturally.


**CORE STRATEGIES (THE "BOT LOGIC")**

1. **Acknowledge & Probe:** Acknowledge their issue simply. Then, ask one natural follow-up question to dig a little deeper into what they actually want.
2. **The Booking Bridge:** Once they answer your probe, frame the human callback as a premium service where the senior team will customize the solution and **make the booking for them**.
3. **The 0120 Anchor:** Always explicitly tell the customer to expect a call from a **"0120"** area code number so they answer it.

**CONVERSATION FLOW**

**Phase 1: Greeting & Discovery**

* **Greeting:** "हेलो, नमस्ते? आई एम नेहा कॉलिंग फ्रॉम यस मैडम। I saw you were checking out some home salon services on our app but didn't complete the booking. Did you face any issue?"
* *Wait for the user to respond, identify their language, and switch your language to match theirs if necessary.*

**Phase 2: Intent Routing (The Probe & Bridge)**
*(Adapt these naturally into English or Hindi. Keep the Indian conversational style. Always probe first, wait for their answer, and then bridge to the callback.)*

* **Branch A: Slots Not Available / Timing Issues**
* *Probe:* "Got it. Were you looking for a morning slot or something in the evening?" *(Wait for reply)*
* *Bridge:* "Right. Evening slots get booked fast. Let me do one thing, I'll have our senior team call you in an hour to find an extra slot and make the booking for you. Should I arrange that?"


* **Branch B: Out of Service Area**
* *Probe:* "Oh, I see. Are you located right in the main city area or slightly towards the outskirts?" *(Wait for reply)*
* *Bridge:* "Got it. Let me have our Area Manager call you to check if we can make a special arrangement today and make the booking for you. Is that fine?"


* **Branch C: Option Overload / Confusion**
* *Probe:* "I know, we have quite a few options! Were you looking for something specific, like a facial or a massage?" *(Wait for reply)*
* *Bridge:* "Nice. Our Korean Facial is actually really good at ₹1350. I'll arrange a quick call with our beauty consultant to suggest the best one and make the booking for you. How does that sound?"


* **Branch D: Service/Product Questions**
* *Probe:* "That's a valid point, we do use sealed kits for hygiene. Do you have any specific skin concerns like sensitivity or tanning?" *(Wait for reply)*
* *Bridge:* "Understood. I can have our Service Specialist call you for two minutes to explain the exact steps and make the booking for you. Should I do that?"


* **Branch E: Price Sensitivity / Nearby Shop Comparison**
* *If the customer says the price is too high or mentions going to a nearby salon:*
* *Probe:* "Got it, I totally understand. Were you looking for a specific budget or maybe a combo package that gives more value?" *(Wait for reply)*
* *Persuade (use naturally, pick what fits):*
  - "I understand, but the best part about Yes Madam is that our professionals are fully trained and certified, and they come to your home with sealed, single-use kits — so hygiene is guaranteed."
  - "Also, we often have app-exclusive offers and combo discounts that local salons can't match. Let me check what's available for you."
  - "Plus, you save time and travel — our experts do everything at your doorstep with premium products."
* *Bridge:* "Let me connect you with our team — they can check for exclusive offers in your budget and make the booking for you. Would you like that?"


* **Branch F: Payment/Technical Failure**
* *Probe:* "Sorry about that. Did the payment link fail, or were you looking for a cash option?" *(Wait for reply)*
* *Bridge:* "I understand. I will ask our payment support team to call you right away with a secure link, or they can simply set up **cash and carry** and make the booking for you. Should I ask them to call?"

* **Branch G: Wants Specific Professional**
* *Probe:* "Right, it's always better to stick with someone you trust. Do you remember the name of the professional who visited last time?" *(Wait for reply)*
* *Bridge:* "Got it. Our managers can check their schedule and try to manually assign them to you while making the booking. Should I arrange that call?"

* **Branch H: Any Other / Uncategorized Issues**
* *Probe:* "Got it. Just to understand better, could you tell me a little more about what exactly happened?" *(Wait for reply)*
* *Bridge:* "Right, that makes sense. Let me do one thing, I will have our specialized support team call you in the next hour to resolve this completely and make the booking for you. Should I arrange that?"

**Phase 3: The Closing**

* **If YES:** "Perfect. I'll mark this on priority. You will get a call from a **0120** number in the next one hour. Have a great day!"
* **If NO/LATER:** "No problem at all. I'll pass your feedback to the team. If you change your mind, you can just book through the app. Have a good day!"

**GUARDRAILS & BOUNDARIES (Strict Adherence)**

* **GREEN ZONE (Must Do):**
* Acknowledge quickly ("Got it", "Right", "I understand").
* Always anchor the "0120" area code before hanging up.
* **Crucial:** Always use the term **"cash and carry"** when referring to paying later (never use the word "advance").
* **Gender-Neutral:** Never assume the customer's gender. Do not use Sir/Ma'am/Madam/bhaiya/didi.

* **RED ZONE (Never Do):**
* **Never pass emojies or any special characters in the response, or tool input / output.**
* **Never say:** "मैं सुनने के लिए हूँ" (I am here to listen).
* **Don't Deny:** Never say "No, we don't do that." Say, "That's a specific request, let me ask my senior team if we can arrange that."
* **Don't Argue Price:** If they mention a competitor, say, "I understand. Our experts can actually check if we can customize a package for you in your budget."
* **Don't Undermine the App:** Frame all solutions as premium, personalized upgrades from the senior team.
""",
        )

    # Removed on_enter since we handle the greeting in entrypoint based on inbound/outbound


    @function_tool(
            name="search_services_summary",
            description="Search for salon services and get a concise list of matching options (Title, Price, Time). Use this FIRST when a user asks about any service or prices to get an overview without reading long descriptions. ALWAYS use short keywords (e.g. 'Rica Wax', 'Facial')."
        )
    async def search_services_summary(
        self,
        query: Annotated[
            str,
            "A short keyword representing the primary service to search for (e.g. 'waxing', 'facial'). NEVER pass full sentences."
        ]
    ):
            """
            Search the pre-loaded JSON services using fuzzy matching for high speed and relevance.
            """
            logger.info(f"Summary search for: {query}")
            if self.call_logger and self.session_data:
                self.call_logger.add_tool_usage(self.session_data, "search_services_summary")
                
            try:
                from thefuzz import process
                
                # Global SERVICES config should be loaded in agent init or globally
                # We extract just the titles to match against
                titles = [s["Service title"] for s in SERVICES]
                
                # Get top 15 fuzzy matches based on title
                best_matches = process.extract(query, titles, limit=15)
                
                results = []
                for match_title, score in best_matches:
                    # Filter out low relevance matches (adjust threshold as needed, e.g., 60)
                    if score < 50:
                        continue
                        
                    # Find the full service dict for this title
                    for s in SERVICES:
                        if s["Service title"] == match_title:
                            # Return highly compressed string to save LLM tokens
                            results.append(f"- {s['Service title']} | {s['Category']} | {s['Basic total cost']} | {s['Approx time']}")
                            break
                            
                if not results:
                    return f"No services found matching '{query}'. Try a shorter or different keyword."
                
                return f"Top matched services for '{query}':\n" + "\n".join(results) + "\n\n(If the user asks for more details on a specific service from this list, use the 'get_service_details' tool with the exact Service title)."
                
            except Exception as e:
                logger.error(f"Error in summary search: {e}")
                return f"Error: Failed to search services. {e}"

    @function_tool(
            name="get_service_details",
            description="Get the full, extensive details for a SPECIFIC service (including benefits, procedure, aftercare, and products used). Use this ONLY after providing a summary, and only when the user explicitly asks for more details about a specific service."
        )
    async def get_service_details(
        self,
        service_title: Annotated[
            str,
            "The EXACT 'Service title' obtained from the search_services_summary tool (e.g., 'Rollover Remedy')."
        ]
    ):
            """
            Fetch the complete text details for a single matched service.
            """
            logger.info(f"Fetching details for: {service_title}")
            if self.call_logger and self.session_data:
                self.call_logger.add_tool_usage(self.session_data, "get_service_details")
                
            try:
                for s in SERVICES:
                    if s["Service title"].lower() == service_title.lower():
                        return f"Full details for {service_title}:\n{s['Full details']}"
                        
                return f"Error: Could not find exact details for '{service_title}'. Please ensure you are using the exact title from the summary list."
            except Exception as e:
                logger.error(f"Error fetching details: {e}")
                return f"Error: Failed to fetch service details. {e}"



server = AgentServer()

@server.rtc_session(agent_name="dummyCode_V2")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # --- PARSE ROOM METADATA ---
    job_metadata = {}
    try:
        job_metadata = json.loads(ctx.job.metadata)
    except Exception:
        pass

    phone_number = job_metadata.get("phone_number")
    campaign_id = job_metadata.get("campaign_id")
    customer_name = job_metadata.get("name", "Customer")
    trunk_id = job_metadata.get("trunk_id", os.getenv("Trunk_ID", "ST_xTmqUEnqWKwQ"))
    room_name = ctx.room.name

    logger.info(f"Job Metadata: phone={phone_number}, campaign={campaign_id}, "
                f"customer={customer_name}, trunk={trunk_id}, room={room_name}")

    # --- START CALL RECORDING (S3) ---
    call_logger = get_call_logger()
    call_uuid = room_name.replace("sip-", "").replace("room-", "")
    # Resolve caller phone: metadata → room name (Inboundcall-<number>) → web fallback
    if phone_number:
        user_phone = phone_number
    elif room_name and room_name.lower().startswith("inboundcall-"):
        user_phone = room_name.split("-", 1)[1]
    else:
        user_phone = f"web_{int(time.time())}"
    
    # Strip LiveKit random suffix (e.g. _+919793370370_RPH2omkdxWkB → +919793370370)
    import re
    phone_match = re.search(r'(\+?\d{10,})', user_phone)
    if phone_match:
        user_phone = phone_match.group(1)
    
    # Initialize logger session
    is_inbound = room_name.lower().startswith("inboundcall") if room_name else False
    session_data = call_logger.log_call_start(call_id=call_uuid, user_phone=user_phone)
    session_data["is_inbound"] = is_inbound
    
    egress_id, recording_filepath = await start_call_recording(ctx, call_uuid)
    if recording_filepath:
        recording_url = build_s3_url(S3_BUCKET, S3_REGION, recording_filepath)
        logger.info(f"Recording URL: {recording_url}")
        session_data["recording_url"] = recording_url

    # --- FINAL LOGGING SHUTDOWN CALLBACK ---
    async def _do_final_logging():
        def _run_sync_logging():
            call_logger.log_call_end(session_data, disconnect_reason="room_disconnected")
            call_log_id = session_data.get("call_log_id")
            
            if session_data.get("_campaign_result_logged"):
                logger.info("Campaign result skipped (already logged for this call)")
                return
                
            if campaign_id and campaign_id != "unknown_campaign" and call_log_id:
                logger.info(f"Campaign result: logging campaign_id={campaign_id}, call_log_id={call_log_id}")
                # Derive status from transcript (inbound = always answered)
                if session_data.get("is_inbound"):
                    answered = "answered"
                else:
                    items = session_data.get("transcript", [])
                    if not items:
                        answered = "unanswered"
                    else:
                        text_content = " ".join(i.get("text", "") for i in items).lower()
                        voicemail_keywords = ["voicemail", "leave a message", "after the beep", "please record"]
                        if any(keyword in text_content for keyword in voicemail_keywords):
                            answered = "voice mail"
                        else:
                            answered = "answered"
                        
                log_outbound_call_result(
                    campaign_id=campaign_id,
                    client_id=os.getenv("MONGO_CLIENT_ID", ""),
                    agent_id=os.getenv("MONGO_AGENT_ID", ""),
                    lead_name=customer_name,
                    lead_number=user_phone or "",
                    status=answered,
                    interest=None,
                    call_log_id=call_log_id,
                )
                session_data["_campaign_result_logged"] = True
                
        # Run the synchronous Mongo/S3 upload blocking-calls in a separate thread
        import asyncio
        await asyncio.to_thread(_run_sync_logging)

    ctx.add_shutdown_callback(_do_final_logging)

    # If a phone number was provided, then place an outbound call
    if phone_number is not None:
        sip_participant_identity = phone_number
        print(f"Outbound call to {phone_number}")
        try:
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                participant_identity=sip_participant_identity,
                wait_until_answered=True,
            ))

            print("call picked up successfully")
        except api.TwirpError as e:
            print(f"error creating SIP participant: {e.message}, "
                  f"SIP status: {e.metadata.get('sip_status_code')} "
                  f"{e.metadata.get('sip_status')}")
            ctx.shutdown()
            return

    session = AgentSession(
        preemptive_generation=True,
        llm=google.realtime.RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        # voice="Autonoe",
        voice="Leda",
        proactivity=True,
        temperature=0.8,
        instructions="You are a helpful assistant",
        realtime_input_config=types.RealtimeInputConfig(
        turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        automatic_activity_detection=types.AutomaticActivityDetection(
        silence_duration_ms=100, 
        prefix_padding_ms=20,
        start_of_speech_sensitivity="START_SENSITIVITY_HIGH",
        end_of_speech_sensitivity="END_SENSITIVITY_HIGH",
    ),
        ),   
    ),)

    from livekit.agents.voice.events import AgentState as VoiceAgentState
    
    session_state = {
        "agent_state": "initializing",
        "last_active": time.time()
    }

    @session.on("agent_state_changed")
    def on_state_changed(state: VoiceAgentState):
        st_str = getattr(state, "new_state", state)
        new_state = str(st_str).lower()
        logger.info(f"[AGENT STATE] {new_state}")
        session_state["agent_state"] = new_state
        session_state["last_active"] = time.time()

    user_speech_start = None
    agent_speech_start = None

    @session.on("user_speech_started")
    def on_user_speech_start(_):
        nonlocal user_speech_start
        user_speech_start = time.time()
        session_state["last_active"] = time.time()  # Block nudge while user is speaking

    @session.on("agent_speech_started")
    def on_agent_speech_start(_):
        nonlocal agent_speech_start
        agent_speech_start = time.time()

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        nonlocal user_speech_start, agent_speech_start
        
        # Reset timer when agent produces content (prevents nudge right after agent speaks)
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) == "assistant" and getattr(item, "content", None):
            session_state["last_active"] = time.time()

        role = getattr(item, 'role', 'unknown')
        text = getattr(item, 'text_content', "")
        interrupted = getattr(item, 'interrupted', False)

        for content in getattr(item, 'content', []):
            if isinstance(content, AudioContent) and content.transcript:
                text = content.transcript
                break

        if text:
            # Print the transcript to the console.
            prefix = f"[{role.upper()}]"
            logger.info(f"{prefix} {text}")
            
            metrics = {}
            current_time = time.time()
            if role == "user" and user_speech_start:
                metrics = {"started_speaking_at": user_speech_start, "stopped_speaking_at": current_time}
                session_state["last_active"] = current_time # Also reset nudge timer on user transcript
                user_speech_start = None
            elif role == "assistant" and agent_speech_start:
                metrics = {"started_speaking_at": agent_speech_start, "stopped_speaking_at": current_time}
                agent_speech_start = None

            call_logger.add_transcript_item(
                session_data,
                role=role,
                content=text,
                interrupted=interrupted,
                metrics=metrics
            )

    # Bind the active session to the logging filter for direct 1008 error recovery
    nudge_filter.active_session = session

    await session.start(
        agent=DefaultAgent(call_logger=call_logger, session_data=session_data),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(
                sync_transcription=True
            ),
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),)
    
    await background_audio.start(room=ctx.room, agent_session=session)

    async def nudge_loop():
        await asyncio.sleep(8)  # Let the agent boot up first
        while True:
            await asyncio.sleep(1)  # Check every 1 second
            
            # If the agent is listening AND 10+ seconds passed without any interim transcripts
            if ("listening" in session_state["agent_state"] and 
                (time.time() - session_state["last_active"]) > 10):
                logger.info("[NUDGE ENGINE] User is silent. Prompting agent to speak.")
                
                # Reset the timer so it doesn't spam
                session_state["last_active"] = time.time()
                
                # Generate the nudge reply with a strict, short timeout to fail faster
                try:
                    # Snappy 1.5-second timeout to quickly catch dead sockets
                    async with asyncio.timeout(1.5):
                        await session.generate_reply(
                            instructions="The user has been silent for a while. Ask them a short, polite question to check if they are still there and if they need help.",
                            allow_interruptions=True,
                        )
                except asyncio.TimeoutError:
                    logger.debug(f"[NUDGE ENGINE] Gemini API dropped the ping (TimeoutError). Triggering rapid backup nudge...")
                    
                    # Separate, inline thread to trigger the recovery immediately
                    async def quick_backup_nudge():
                        await asyncio.sleep(0.5)
                        
                        # Abort the backup if the original ping managed to successfully transition the agent
                        if "listening" not in session_state["agent_state"]:
                            return
                            
                        try:
                            await session.generate_reply(
                                instructions="The user has been silent for a while. Ask them a short, polite question to check if they are still there and if they need help.",
                                allow_interruptions=True,
                            )
                        except Exception:
                            pass

                    asyncio.create_task(quick_backup_nudge())
                    # Reset the main timer back to 0 so the primary loop doesn't double-fire
                    session_state["last_active"] = time.time()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # Generic catch-all prevents breaking the heartbeat if the API natively resets connection sockets natively
                    if "1008" in str(e):
                        session_state["last_active"] = time.time()
                    else:
                        logger.error(f"[NUDGE ENGINE] Failed to generate reply: {e}")
                        session_state["last_active"] = time.time() # Reset timer on general failure

    # Start the non-blocking background task
    asyncio.create_task(nudge_loop())

    # Add this guard to ensure the agent only speaks first in an inbound scenario.
    await session.generate_reply(
            instructions="""Greet user with \" हेलो, नमस्ते आई एम नेहा कॉलिंग फ्रॉम यस मैडम। \"""",
            allow_interruptions=True,
        )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="dummyCode_V2",
            port=int(os.getenv("PORT", 8080)),
        )
    )



# - You can use tools to get the information related too.
