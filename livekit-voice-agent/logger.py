"""
Call Logger Service
Pushes call logs to MongoDB for tracking and analytics.
Uploads transcripts to S3 in LiveKit format.
Synchronous logging on disconnect to ensure data is captured.
"""

import time
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import quote

try:
    from pymongo import ReturnDocument
except ImportError:
    ReturnDocument = None  # type: ignore

# Specialized imports
import google.genai as genai
from google.genai import types

logger = logging.getLogger("call-logger")

# Lazy import to avoid startup blocking
pymongo = None
minio = None

def _get_pymongo():
    global pymongo
    if pymongo is None:
        try:
            import pymongo as _pymongo
            pymongo = _pymongo
        except ImportError:
            logger.warning("pymongo not installed. MongoDB logging disabled.")
    return pymongo

def _get_minio():
    global minio
    if minio is None:
        try:
            from minio import Minio as _Minio
            minio = _Minio
        except ImportError:
            logger.warning("minio not installed. S3 transcript upload disabled.")
    return minio


class CallLogger:
    """
    Call logger that pushes to MongoDB.
    Synchronous logging ensures data is captured before process exits.
    """
    
    def __init__(self):
        # MongoDB config
        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_client = None
        self.mongo_db = None
        self.call_logs_collection = None
        self._mongo_initialized = False
    
    def _init_mongodb(self):
        """Initialize MongoDB client lazily"""
        if self._mongo_initialized:
            return
            
        self._mongo_initialized = True
        
        if not self.mongo_uri:
            print("âŒ MONGO_URI is NOT set in environment!")
            logger.warning("MONGO_URI not set. MongoDB logging disabled.")
            return
            
        print(f"ðŸ“¡ Initializing MongoDB connection with URI: {self.mongo_uri[:20]}...")
        mongo = _get_pymongo()
        if not mongo:
            return
            
        try:
            self.mongo_client = mongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000, maxPoolSize=2, minPoolSize=0)
            # Extract database name from URI or use default
            # URI format: mongodb+srv://user:pass@cluster/dbname?options
            db_name = self.mongo_uri.split('/')[-1].split('?')[0] if '/' in self.mongo_uri else "voiceai"
            if not db_name or db_name == "":
                db_name = "voiceai"
            self.mongo_db = self.mongo_client[db_name]
            self.call_logs_collection = self.mongo_db["calllogs"]
            logger.info(f"ðŸ“Š MongoDB Logger initialized: {db_name}.calllogs")
        except Exception as e:
            logger.error(f"Failed to init MongoDB: {e}")
            self.mongo_client = None
    
    def log_call_start(self, call_id: str, user_phone: str, is_inbound: bool = False) -> Dict[str, Any]:
        """
        Initialize session data at call start.
        Returns a session_data dict to be passed to log_call_end.
        """
        session_data = {
            "call_id": call_id,
            "user_phone": user_phone,
            "start_time": datetime.utcnow().isoformat() + "Z",
            "start_ts": time.time(),
            "transcript": [],  # Legacy format for backward compatibility
            "transcript_items": [],  # LiveKit format items
            "tools_used": [],
            "disconnect_reason": "unknown",
            "transcript_uri": "",
            "recording_url": "",
            "is_inbound": is_inbound
        }
        # Create or update an initial MongoDB record so the call appears
        # in analytics even if the worker crashes mid-call.
        try:
            self._upsert_call_log(session_data, hand_off=False)
        except Exception as e:
            logger.warning(f"MongoDB incremental upsert (start) failed: {e}")
        return session_data
    
    def add_transcript(self, session_data: Dict, speaker: str, text: str):
        """Add a transcript entry to session data (legacy format)"""
        if "transcript" not in session_data:
            session_data["transcript"] = []
        session_data["transcript"].append({
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    def add_transcript_item(self, session_data: Dict, role: str, content: str, 
                           interrupted: bool = False, confidence: float = None,
                           metrics: Dict = None):
        """
        Add a transcript item in LiveKit format.
        """
        if "transcript_items" not in session_data:
            session_data["transcript_items"] = []
        
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        item = {
            "id": item_id,
            "type": "message",
            "role": role,
            "content": [content] if content else [],
            "interrupted": interrupted,
            "extra": {},
            "metrics": metrics or {}
        }
        
        if role == "user" and confidence is not None:
            item["transcript_confidence"] = confidence
        
        session_data["transcript_items"].append(item)
        
        # Also add to legacy format for backward compatibility
        speaker = "User" if role == "user" else "Agent"
        self.add_transcript(session_data, speaker, content)

        # Incremental MongoDB upsert so we don't lose data on crashes
        try:
            self._upsert_call_log(session_data, hand_off=False)
        except Exception as e:
            logger.warning(f"MongoDB incremental upsert (transcript) failed: {e}")
    
    def add_handoff_item(self, session_data: Dict, agent_id: str = "human_agent"):
        """Add a handoff item to transcript"""
        if "transcript_items" not in session_data:
            session_data["transcript_items"] = []
        
        item = {
            "id": f"item_{uuid.uuid4().hex[:12]}",
            "type": "agent_handoff",
            "new_agent_id": agent_id
        }
        session_data["transcript_items"].append(item)

        # Incremental upsert marking that a handoff has occurred
        try:
            self._upsert_call_log(session_data, hand_off=True)
        except Exception as e:
            logger.warning(f"MongoDB incremental upsert (handoff) failed: {e}")
    
    def _upload_transcript_to_s3(self, session_data: Dict) -> str:
        """
        Upload transcript JSON to S3 in LiveKit format using Minio.
        Returns the S3 URL or empty string if upload fails.
        """
        Minio = _get_minio()
        if not Minio:
            logger.warning("minio not available. Skipping transcript upload.")
            return ""
        
        # Get S3 config from environment
        bucket = os.getenv("BUCKET", "caller-recordings")
        region = os.getenv("REGION", "ap-south-1")
        access_key = os.getenv("ACCESS_KEY")
        secret_key = os.getenv("SECRET")
        
        if not access_key or not secret_key:
            logger.warning("S3 credentials not set. Skipping transcript upload.")
            return ""
        
        try:
            import io
            # Create Minio S3 client
            s3_client = Minio(
                f"s3.{region}.amazonaws.com",
                access_key=access_key,
                secret_key=secret_key,
                secure=True
            )
            
            # Build LiveKit format transcript
            transcript_json = {
                "items": session_data.get("transcript_items", [])
            }
            
            # Generate S3 key
            call_id = session_data.get("call_id", str(uuid.uuid4()))
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"transcripts/{call_id}/transcript_{timestamp}.json"
            
            data = json.dumps(transcript_json, indent=2).encode("utf-8")
            
            # Upload to S3
            s3_client.put_object(
                bucket,
                s3_key,
                io.BytesIO(data),
                len(data),
                content_type="application/json"
            )
            
            # Build URL with proper encoding
            encoded_key = quote(s3_key, safe='/')
            transcript_url = f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"
            logger.info(f"ðŸ“„ Transcript uploaded to: {transcript_url}")
            return transcript_url
            
        except Exception as e:
            logger.error(f"â Œ Failed to upload transcript to S3: {e}")
            return ""
    
    def add_tool_usage(self, session_data: Dict, tool_name: str, result: str = ""):
        """Track tool usage in session data"""
        if "tools_used" not in session_data:
            session_data["tools_used"] = []
        session_data["tools_used"].append({
            "tool": tool_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result_preview": result[:100] if result else ""
        })

        return

    def _extract_entities(self, transcript_text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from transcript using Gemini Flash Lite.
        """
        if not transcript_text:
            return []

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not set. Skipping entity extraction.")
            return []

        try:
            client = genai.Client(api_key=google_api_key)
            
            prompt = f"""
            Extract the following entities from the call transcript of a Yes Madam cart recovery call.
            Return a JSON list of key-value pairs.
            
            Entities to extract:
            - Reason for Dropoff : The exact reason the customer didn't complete their booking (e.g. "slots not available", "too expensive", "confused by options", "payment failed", "out of service area"). Be specific and concise.
            - Agreed for Callback : ENUM (YES / NO) - Did the customer agree to receive a callback from the senior team?
            - Preferred Callback Time : The specific time/slot the customer requested for the callback (e.g. "evening 6pm", "tomorrow morning"). If not mentioned, put "Not specified".
            - Special Request : Any special request or preference the customer mentioned (e.g. "wants same beautician as last time", "needs hypoallergenic products"). If none, put "None".
            
            Transcript:
            {transcript_text}
            
            Return ONLY a JSON list, for example:
            [
              {{"key": "Reason for Dropoff", "value": "slots were not available for preferred time"}},
              {{"key": "Agreed for Callback", "value": "YES"}},
              {{"key": "Preferred Callback Time", "value": "evening around 6pm"}},
              {{"key": "Special Request", "value": "None"}}
            ]
            """
            
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                entities = json.loads(response.text)
                if isinstance(entities, list):
                    return entities
            return []
            
        except Exception as e:
            logger.error(f"âŒ Entity extraction failed: {e}")
            return []
    
    def log_call_end(self, session_data: Dict[str, Any], disconnect_reason: str = "unknown", hand_off: bool = False):
        """
        Pushes call summary to MongoDB.
        """
        # Calculate duration and times
        start_ts = session_data.get("start_ts", time.time())
        duration = int(time.time() - start_ts)
        start_time_str = session_data.get("start_time", datetime.utcnow().isoformat() + "Z")
        end_time_str = datetime.utcnow().isoformat() + "Z"
        
        # Convert transcript list to string (for reference)
        transcript_text = ""
        if session_data.get("transcript"):
            transcript_text = "\n".join([
                f"{t['speaker']}: {t['text']}" 
                for t in session_data.get("transcript", [])
            ])
        
        # Upload transcript to S3 and get URL
        transcript_uri = self._upload_transcript_to_s3(session_data)
        session_data["transcript_uri"] = transcript_uri
        
        # --- Entity Extraction (Post-Call) ---
        entities = self._extract_entities(transcript_text)
        session_data["entities"] = entities
        session_data["disconnect_reason"] = disconnect_reason

        # --- Status (answered / voice mail / unanswered) ---
        if session_data.get("is_inbound"):
            answered = "answered"
        else:
            items = session_data.get("transcript", [])
            if not items:
                answered = "unanswered"
            else:
                text_content = " ".join(t.get("text", "") for t in items).lower()
                voicemail_keywords = ["voicemail", "leave a message", "after the beep", "please record"]
                if any(keyword in text_content for keyword in voicemail_keywords):
                    answered = "voice mail"
                else:
                    answered = "answered"
        session_data["status"] = answered
        print(f"ðŸ“Š Call status derived for {session_data.get('call_id')}: {answered}")

        # Final upsert with ALL data; get call log _id for campaign results
        doc_id = self._upsert_call_log(session_data, hand_off=hand_off, entities=entities)
        session_data["call_log_id"] = str(doc_id) if doc_id else None

    def _upsert_call_log(self, session_data: Dict, hand_off: bool = False, entities: List[Dict] = None) -> Optional[Any]:
        """
        Incrementally upsert a MongoDB document for the in-progress call.
        Returns the document _id (ObjectId) so callers can use it e.g. for campaign results.
        """
        self._init_mongodb()

        if self.call_logs_collection is None:
            return None

        call_id = session_data.get("call_id", "")
        if not call_id:
            print("âš ï¸ Skipping MongoDB upsert: call_id is empty")
            return None

        # Import ObjectId for MongoDB
        try:
            from bson import ObjectId
        except ImportError:
            logger.error("bson not available. Cannot create ObjectIds.")
            return None

        # Parse phone number and IDs
        user_phone = session_data.get("user_phone", "")
        if user_phone.startswith("sip_"):
            user_phone = user_phone[4:]

        agent_id = os.getenv("MONGO_AGENT_ID", "")
        client_id = os.getenv("MONGO_CLIENT_ID", "")

        # Compute timing snapshot
        start_ts = session_data.get("start_ts", time.time())
        duration = int(time.time() - start_ts)
        start_time_str = session_data.get("start_time", datetime.utcnow().isoformat() + "Z")
        end_time_str = datetime.utcnow().isoformat() + "Z"

        disconnect_reason = session_data.get("disconnect_reason", "in_progress")

        # Simple outcome for in-progress upserts
        if disconnect_reason in ["agent_hangup", "conversation_complete", "transfer_to_human", "timeout"]:
            call_ended_by = "AGENT"
        else:
            call_ended_by = "User"

        if disconnect_reason == "transfer_to_human":
            call_outcome = "escalated"
        elif disconnect_reason in ["user_hangup", "agent_hangup", "conversation_complete", "room_disconnected"]:
            call_outcome = "completed"
        elif disconnect_reason == "timeout":
            call_outcome = "no_response"
        else:
            call_outcome = "in_progress"

        # Use passed entities or existing ones in session
        final_entities = entities if entities is not None else session_data.get("entities", [])

        shared_state = {
            "call_id": session_data.get("call_id", ""),
            "user_phone": user_phone,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "duration": duration,
            "transcript": session_data.get("transcript", []),
            "entities": final_entities,
            "disconnect_reason": disconnect_reason,
            "human_hand_off": hand_off,
        }

        status = session_data.get("status")
        if status is not None:
            shared_state["status"] = status

        now = datetime.utcnow()
        update_doc = {
            "$set": {
                "call_id": call_id,
                "transcript_uri": session_data.get("transcript_uri", ""),
                "summary_uri": "",
                "recording_uri": session_data.get("recording_url", ""),
                "call_duration": duration,
                "call_start_time": datetime.fromisoformat(start_time_str.replace("Z", "+00:00")),
                "call_end_time": datetime.fromisoformat(end_time_str.replace("Z", "+00:00")),
                "agentId": ObjectId(agent_id) if agent_id else None,
                "clientId": ObjectId(client_id) if client_id else None,
                "customer_phone_number": user_phone,
                "call_type": "inbound" if session_data.get("is_inbound") else "outbound",
                "agent_phone_number": os.getenv("AGENT_PHONE_NUMBER", ""),
                "shared_state": shared_state,
                "hand_off": hand_off,
                "entity_result": final_entities,
                "callEndedBy": call_ended_by,
                "call_outcome1": disconnect_reason,
                "call_outcome": call_outcome,
                "status": status,
                "updatedAt": now,
            },
            "$setOnInsert": {
                "createdAt": now,
            },
        }

        try:
            print(f"ðŸ“¡ Pushing to MongoDB: {call_id} | Outcome: {call_outcome}")
            print(f"ðŸ“ Transcript count: {len(session_data.get('transcript', []))}")
            print(f"ðŸ“Š Mongo status field for {call_id}: {status}")
            if final_entities:
                print(f"ðŸ’Ž AI Entities Extracted: {json.dumps(final_entities, indent=2)}")
            
            # Use find_one_and_update to get the document _id for campaign results linking
            if ReturnDocument is not None:
                updated = self.call_logs_collection.find_one_and_update(
                    {"call_id": call_id},
                    update_doc,
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )
                return updated.get("_id") if updated else None
            else:
                self.call_logs_collection.update_one(
                    {"call_id": call_id},
                    update_doc,
                    upsert=True,
                )
                return None
        except Exception as e:
            logger.error(f"âŒ MongoDB incremental upsert FAILED: {e}")
            return None
        
    def _push_to_mongodb(self, session_data: Dict, disconnect_reason: str, duration: int, 
                         start_time: str, end_time: str, hand_off: bool, transcript_text: str, entities: List[Dict] = None):
        """Push call log to MongoDB"""
        self._init_mongodb()
        
        if self.call_logs_collection is None:
            logger.warning("MongoDB not configured. Skipping.")
            return
        
        # Import ObjectId for MongoDB
        try:
            from bson import ObjectId
        except ImportError:
            logger.error("bson not available. Cannot create ObjectIds.")
            return
        
        # Parse phone number
        user_phone = session_data.get("user_phone", "")
        if user_phone.startswith("sip_"):
            user_phone = user_phone[4:]
        
        # Get ObjectIds from environment
        agent_id = os.getenv("MONGO_AGENT_ID", "")
        client_id = os.getenv("MONGO_CLIENT_ID", "")
        
        # Determine callEndedBy
        if disconnect_reason in ["agent_hangup", "conversation_complete", "transfer_to_human", "timeout"]:
            call_ended_by = "AGENT"
        else:
            call_ended_by = "User"
        
        # Determine call outcome
        if disconnect_reason == "transfer_to_human":
            call_outcome = "escalated"
        elif disconnect_reason in ["user_hangup", "agent_hangup", "conversation_complete"]:
            call_outcome = "completed"
        elif disconnect_reason == "timeout":
            call_outcome = "no_response"
        else:
            call_outcome = "unknown"
        
        # Build shared_state (initially with passed/existing entities)
        current_entities = entities or session_data.get("entities", [])
        shared_state = {
            "call_id": session_data.get("call_id", ""),
            "user_phone": user_phone,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "transcript": session_data.get("transcript", []),
            "entities": current_entities,
            "disconnect_reason": disconnect_reason,
            "human_hand_off": hand_off,
        }
        
        # --- Entity Extraction ---
        new_entities = self._extract_entities(transcript_text)
        if new_entities:
            shared_state["entities"] = new_entities
            current_entities = new_entities
        
        # Prepare MongoDB document
        mongo_doc = {
            "transcript_uri": session_data.get("transcript_uri", ""),
            "summary_uri": "",
            "recording_uri": session_data.get("recording_url", ""),
            "call_duration": duration,
            "call_start_time": datetime.fromisoformat(start_time.replace("Z", "+00:00")),
            "call_end_time": datetime.fromisoformat(end_time.replace("Z", "+00:00")),
            "agentId": ObjectId(agent_id) if agent_id else None,
            "clientId": ObjectId(client_id) if client_id else None,
            "customer_phone_number": user_phone,
            "call_type": "inbound" if session_data.get("is_inbound") else "outbound",
            "agent_phone_number": os.getenv("AGENT_PHONE_NUMBER", ""),
            "shared_state": shared_state,
            "hand_off": hand_off,
            "entity_result": current_entities,
            "callEndedBy": call_ended_by,
            "creditsDeducted": False,
            "call_outcome1": disconnect_reason,
            "call_outcome": call_outcome,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
        }
        
        try:
            result = self.call_logs_collection.insert_one(mongo_doc)
            print(f"âœ… MongoDB: {user_phone} | Duration: {duration}s | _id={result.inserted_id}")
        except Exception as e:
            print(f"âŒ MongoDB FAILED: {e}")


# Global singleton instance
_call_logger: Optional[CallLogger] = None

def get_call_logger() -> CallLogger:
    """Get the global call logger instance"""
    global _call_logger
    if _call_logger is None:
        _call_logger = CallLogger()
    return _call_logger


_db_client = None
_db = None


def get_db():
    """Get MongoDB database (same URI as CallLogger) for campaign results etc."""
    global _db_client, _db
    if _db is not None:
        return _db
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None
    mongo = _get_pymongo()
    if not mongo:
        return None
    try:
        _db_client = mongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, maxPoolSize=2, minPoolSize=0)
        db_name = mongo_uri.split("/")[-1].split("?")[0] if "/" in mongo_uri else "voiceai"
        if not db_name:
            db_name = "voiceai"
        _db = _db_client[db_name]
        return _db
    except Exception as e:
        logger.error(f"get_db failed: {e}")
        return None


def log_outbound_call_result(
    campaign_id: str,
    client_id: str,
    agent_id: str,
    lead_name: str,
    lead_number: str,
    status: str = "unanswered",
    interest: Optional[bool] = None,
    call_log_id: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Log outbound call result to MongoDB CampaignResult collection.
    Links to call log via call_log_id (calllogs document _id).
    """
    try:
        from bson import ObjectId
    except ImportError:
        logger.error("bson not available for log_outbound_call_result")
        return False

    db = get_db()
    if db is None:
        logger.warning("MongoDB connection not available for campaign result")
        return False

    collection = db.campaignresults
    campaign_result_data = {
        "campaignId": ObjectId(campaign_id) if campaign_id else None,
        "clientId": ObjectId(client_id) if client_id else None,
        "agentId": ObjectId(agent_id) if agent_id else None,
        "status": status,
        "leadName": lead_name,
        "leadNumber": lead_number,
        "interest": interest,
        "callLogId": ObjectId(call_log_id) if call_log_id else None,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
    }
    campaign_result_data.update(kwargs)

    try:
        collection.insert_one(campaign_result_data)
        logger.info(f"Outbound call result logged: campaign={campaign_id}, lead={lead_name}, status={status}")
        return True
    except Exception as e:
        logger.error(f"Failed to log outbound call result: {e}")
        return False