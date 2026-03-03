import os
import json
import logging
import asyncio
import time
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
    JobProcess,
    cli,
    inference,
    room_io,
    llm,
)
from livekit.plugins import (
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import google, deepgram, silero
from google.genai import types
from google.genai.errors import APIError
from livekit.agents.beta.tools import EndCallTool

logger = logging.getLogger("agent-dummyCode")

# Mask Livekit's noisy internal Realtime errors that leak past try-except blocks
class SuppressLivekitNudgeTraceback(logging.Filter):
    def __init__(self):
        super().__init__()
        self.active_session = None

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            msg = record.getMessage()
            if "Error in _realtime_reply_task" in msg:
                return False
            if "error in receive task: 1008" in msg:
                # If we have an active session, use this logger intercept to trigger an emergency nudge
                if self.active_session:
                    logger.warning("[LOGGER RECOVERY] Google Gemini forcibly closed the socket (1008 Policy Violation). Triggering rapid backup nudge in 2 seconds...")
                    
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
                        pass # No running event loop
                        
                return False
        return True

nudge_filter = SuppressLivekitNudgeTraceback()
logging.getLogger("livekit.agents").addFilter(nudge_filter)
logging.getLogger("livekit.plugins.google").addFilter(nudge_filter)

load_dotenv(".env.local")

# Load services once at startup for high-speed access
try:
    with open(os.path.join(os.path.dirname(__file__), "services.json"), 'r', encoding='utf-8') as f:
        SERVICES = json.load(f)
    logger.info(f"Loaded {len(SERVICES)} services into memory.")
except Exception as e:
    logger.error(f"Failed to load services.json: {e}")
    SERVICES = []
    
class DefaultAgent(Agent):
    def __init__(self) -> None:
        end_call_tool = EndCallTool(
            extra_description="Only end the call after confirming the customer's issue is resolved or they want to hang up.",
            delete_room=True,
            end_instructions="Thank the customer for their time and wish them a good day.",
        )
        super().__init__(
            tools=end_call_tool.tools,
            instructions="""You are Neha, an AI Assistant for Yes Madam home salon services (25F, Indian, polite, bilingual Hindi/English).
Mission: Call users who abandoned their cart, find their issue, probe once, and pitch a callback from a senior expert who will finalize the booking.

RULES:
- Always use an Indian accent. Start with "हेलो, नमस्ते?".
- If user speaks Hindi, reply in Devanagari script. Never echo back what they said. Talk concisely (under 3 sentences).
- Only tell prices if explicitly asked. Never fabricate answers. Never use emojis.
- TOOL FAILURE: If a tool returns an error/empty, immediately apologize and ask if they need anything else. Do not stay silent.
- TOOLS: Use `search_services_summary` first. Pitch 1-2 options. Only use `get_service_details` if asked for deep specifics.

ROUTING LOGIC:
A. Timing: Pitch an extra slot via senior team.
B. Area: Pitch a special arrangement via Area Manager.
C. Confusion: Suggest a specific service (like Korean Facial ₹1350) via beauty consultant.
D. Questions: Reassure about hygiene/kits. Offer a Service Specialist call.
E. Price: Pitch loyalty team for over-the-call offers.
F. Payment: Pitch secure link or 'cash and carry' (never say advance) via support team.
G. Specific pro: Pitch manager to manually assign them.
H. Other: Ask for details, pitch specialized support team resolution.

CLOSING:
- If YES: "Perfect. You will get a call from a 0120 number in the next one hour. Have a great day!"
- If NO: "No problem. Book through the app anytime. Have a good day!"
""",
        )

    # Removed on_enter since we handle the greeting in entrypoint based on inbound/outbound


    @function_tool(
            name="search_services_summary",
            description="Get a quick list of matching salon services (Title, Price, Time) using short keywords (e.g. 'waxing', 'facial'). Do not use full sentences."
        )
    async def search_services_summary(
        self,
        query: Annotated[str, "Short keyword (e.g. 'waxing'). NO sentences."]
    ):
            """
            Search the pre-loaded JSON services using fuzzy matching for high speed and relevance.
            """
            logger.info(f"Summary search for: {query}")
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
            description="Get extensive text details for ONE specific service. Use ONLY after search_services_summary if the user asks for deep specifics."
        )
    async def get_service_details(
        self,
        service_title: Annotated[str, "Exact 'Service title' from summary."]
    ):
            """
            Fetch the complete text details for a single matched service.
            """
            logger.info(f"Fetching details for: {service_title}")
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
    
    # If a phone number was provided, then place an outbound call
    try:
        dial_info = json.loads(ctx.job.metadata)
        phone_number = dial_info.get("phone_number")
    except Exception:
        phone_number = None

    # The participant's identity can be anything you want, but this example uses the phone number itself
    if phone_number is not None:
        sip_participant_identity = phone_number
        print(f"Outbound call to {phone_number}")
        # The outbound call will be placed after this method is executed
        try:
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                # This ensures the participant joins the correct room
                room_name=ctx.room.name,
                # This is the outbound trunk ID to use (i.e. which phone number the call will come from)
                # You can get this from LiveKit CLI with `lk sip outbound list`
                sip_trunk_id=os.getenv("Trunk_ID", "ST_gL3qKS6ftTEi"),
                # The outbound phone number to dial and identity to use
                sip_call_to=phone_number,
                participant_identity=sip_participant_identity,
                # This will wait until the call is answered before returning
                wait_until_answered=True,
            )

            )

            print("call picked up successfully")
        except api.TwirpError as e:
            print(f"error creating SIP participant: {e.message}, "
                  f"SIP status: {e.metadata.get('sip_status_code')} "
                  f"{e.metadata.get('sip_status')}")
            ctx.shutdown()
            return

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Leda",
            temperature=0.8,
            instructions="You are a helpful assistant",
        ),
        preemptive_generation=True,
        turn_detection=MultilingualModel(),
    )

    from livekit.agents.voice.events import AgentState as VoiceAgentState
    from livekit.agents.llm.realtime import RealtimeError
    
    session_state = {
        "agent_state": "initializing",
        "last_active": time.time()
    }

    @session.on("agent_state_changed")
    def on_state_changed(state: VoiceAgentState):
        st_str = getattr(state, "new_state", state)
        new_state = str(st_str).lower()
        session_state["agent_state"] = new_state
        session_state["last_active"] = time.time()


    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        # Reset the timer whenever ANY speech is detected (final or interim)
        if getattr(ev, "transcript", None):
            session_state["last_active"] = time.time()

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        # The realtime model pushes conversation items; we catch the assistant side here
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) == "assistant" and getattr(item, "content", None):
            session_state["last_active"] = time.time()

    # Bind the active session to the custom Logger Filter so it can catch 1008 drops
    nudge_filter.active_session = session

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(
                sync_transcription=True
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),)
    
    await background_audio.start(room=ctx.room, agent_session=session)

    async def nudge_loop():
        await asyncio.sleep(5) # Let the agent boot up first
        while True:
            await asyncio.sleep(1) # Check every 1 second
            
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
                except (TimeoutError, RealtimeError) as e:
                    logger.debug(f"[NUDGE ENGINE] Gemini API dropped the ping ({type(e).__name__}). Triggering rapid backup nudge...")
                    
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
                except APIError:
                    # Socket dropped; the global @session.on("error") handler will catch this and trigger the 2s recovery
                    session_state["last_active"] = time.time()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[NUDGE ENGINE] Failed to generate reply: {e}")
                    session_state["last_active"] = time.time() # Reset timer on general failure

    # Start the non-blocking background task
    asyncio.create_task(nudge_loop())

    # Add this guard to ensure the agent only speaks first in an inbound scenario.
    if phone_number is None:
        await session.generate_reply(
            instructions="""Greet user with \" हेलो, नमस्ते आई एम नेहा कॉलिंग फ्रॉम यस मैडम। \"""",
            allow_interruptions=True,
        )

if __name__ == "__main__":
    cli.run_app(server)



# - You can use tools to get the information related too.
