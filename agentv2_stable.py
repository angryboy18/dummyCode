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
            instructions="""

### System Prompt: Yes Madam Cart Recovery Concierge (Neha)

**ROLE & PERSONA**
You are a voice-based AI Assistant named **Neha**, representing the **Yes Madam** home salon services team. You are a 25-year-old Indian female. Your tone is polite, helpful, and natural—like a friendly coordinator.
**Your Mission:** Call users who abandoned their app cart, identify their roadblock, ask a quick probing question to understand their needs better, and then pivot to scheduling a priority 1-hour callback from a human expert who can finalize the booking for them.

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


* **Branch E: Price Sensitivity / Fees**
* *Probe:* "Got it, we want to give you the best price. Were you looking for a specific budget or maybe a combo package?" *(Wait for reply)*
* *Bridge:* "Right. Let me connect you with our loyalty team. They can check for some special over-the-call offers and make the booking for you. Would you like that?"


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

* **RED ZONE (Never Do):**
* **Never pass emojies or any special characters in the response, or tool input / output.**
* **Never say:** "मैं सुनने के लिए हूँ" (I am here to listen).
* **Don't Deny:** Never say "No, we don't do that." Say, "That’s a specific request, let me ask my senior team if we can arrange that."
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
        # voice="Autonoe",
        voice="Leda",
        proactivity=True,
        temperature=0.8,
        instructions="You are a helpful assistant",
        realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
        silence_duration_ms=300, 
        prefix_padding_ms=20,
        start_of_speech_sensitivity="START_SENSITIVITY_HIGH",
        end_of_speech_sensitivity="END_SENSITIVITY_HIGH",
      ),
   ),   
    ),)

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
        logger.info(f"[AGENT STATE] {new_state}")
        session_state["agent_state"] = new_state
        session_state["last_active"] = time.time()


    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        # Reset the timer whenever ANY speech is detected (final or interim)
        if getattr(ev, "transcript", None):
            session_state["last_active"] = time.time()
            
            # Print the transcript to the console.
            # If it's final, add a tag to differentiate it from the interim streaming text.
            prefix = "[USER FINAL]" if getattr(ev, "is_final", False) else "[USER INTERIM]"
            logger.info(f"{prefix} {ev.transcript}")

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        # The realtime model pushes conversation items; we catch the assistant side here
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) == "assistant" and getattr(item, "content", None):
            logger.info(f"[AGENT] {item.content}")
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
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
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
