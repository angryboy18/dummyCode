import os
import json
import logging
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
from livekit.agents.beta.tools import EndCallTool

logger = logging.getLogger("agent-dummyCode")

load_dotenv(".env.local")

    
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
- You can use tools to get the information related too
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
- Below privded script are not mandatory to follow, you can change them as per the conversation.
- Conversation should feel natural and human-like.

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
* Know the Hero prices: Korean Facial (₹1350), Rica Waxing (₹750), Platform Fee (~₹100).
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
            name="search_services",
            description="Search for salon services, waxing, message services and their prices, and approximate time taking. Use this when the user asks about specific services or their prices. STRICTLY use short keywords ONLY (e.g. 'Rica Wax', 'Facial', 'Manicure'). Never submit questions or long phrases."
        )
    async def search_services(
        self,
        query: Annotated[
            str,
            "A single, short keyword representing the primary service or category to search for (e.g. 'waxing', 'threading', 'facial', 'spa'). NEVER pass full sentences or multiple concepts at once."
        ]
    ):
            """
            Search the services markdown file for matching entries, specifically looking in the Category/Subcategory fields.
            """
            logger.info(f"Searching services for: {query}")
            try:
                # Assuming services.md is in the same directory as agent.py or we use the absolute path for safety
                md_path = os.path.join(os.path.dirname(__file__), "services.md")
                if not os.path.exists(md_path):
                    return "Error: Services file not found."

                results = []
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    services = content.split("## ")
                    
                    for service in services:
                        if not service.strip(): continue
                        
                        # Extract the Category line to see if the query matches the subcategory specifically
                        lines = service.strip().split('\n')
                        category_line = ""
                        for line in lines:
                            if line.startswith("- Category:"):
                                category_line = line.lower()
                                break
                        
                        # Check if query matches the category exactly or is in the general service body
                        if query.lower() in category_line:
                            # Higher priority match if it's in the category
                            results.insert(0, "## " + service.strip())
                        elif query.lower() in service.lower():
                            results.append("## " + service.strip())
                
                if not results:
                    return f"No services found matching '{query}'."
                
                # Limit results to avoid context overflow if query is too broad, since each service entry is now very detailed
                if len(results) > 3:
                    return f"Found {len(results)} services matching the category or keyword. Here are the top 3 best matching services:\n\n" + "\n\n".join(results[:3])
                
                return "\n\n".join(results)
            except Exception as e:
                logger.error(f"Error searching services: {e}")
                return f"Error: Failed to search services. {e}"



server = AgentServer()

@server.rtc_session(agent_name="dummyCode")
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
        allow_interruptions=True,
        llm=google.realtime.RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        # voice="Autonoe",
        voice="Leda",
        proactivity=True,
        temperature=0.8,
        instructions="You are a helpful assistant",),
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),)
    
    await background_audio.start(room=ctx.room, agent_session=session)

    # Add this guard to ensure the agent only speaks first in an inbound scenario.
    if phone_number is None:
        await session.generate_reply(
            instructions="""Greet user with \" हेलो, नमस्ते आई एम नेहा कॉलिंग फ्रॉम यस मैडम। \"""",
            allow_interruptions=False,
        )

if __name__ == "__main__":
    cli.run_app(server)



