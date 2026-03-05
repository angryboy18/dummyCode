#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# GCP Cloud Run Deployment Script for LiveKit Voice Agent
# ─────────────────────────────────────────────────────────────────
# Prerequisites:
#   1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
#   2. Authenticate: gcloud auth login
#   3. Set project: gcloud config set project YOUR_PROJECT_ID
#   4. Enable APIs: gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration (EDIT THESE) ──
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE_NAME="livekit-voice-agent"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}"

# ── Step 1: Build & Push using Cloud Build ──
echo "🔨 Building and pushing Docker image..."
gcloud builds submit \
  --tag "${IMAGE_NAME}" \
  --timeout=1200 \
  --project="${PROJECT_ID}"

# ── Step 2: Deploy to Cloud Run ──
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --cpu=2 \
  --memory=2Gi \
  --timeout=3600 \
  --min-instances=1 \
  --max-instances=5 \
  --concurrency=4 \
  --port=8081 \
  --no-allow-unauthenticated \
  --set-env-vars="LIVEKIT_URL=${LIVEKIT_URL}" \
  --set-env-vars="LIVEKIT_API_KEY=${LIVEKIT_API_KEY}" \
  --set-env-vars="LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}" \
  --set-env-vars="GOOGLE_API_KEY=${GOOGLE_API_KEY}" \
  --set-env-vars="DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}" \
  --set-env-vars="Trunk_ID=${Trunk_ID}" \
  --set-env-vars="MONGO_URI=${MONGO_URI}" \
  --set-env-vars="MONGO_AGENT_ID=${MONGO_AGENT_ID}" \
  --set-env-vars="MONGO_CLIENT_ID=${MONGO_CLIENT_ID}" \
  --set-env-vars="ACCESS_KEY=${ACCESS_KEY}" \
  --set-env-vars="SECRET=${SECRET}" \
  --set-env-vars="REGION=${AWS_REGION:-ap-south-1}" \
  --set-env-vars="BUCKET=${BUCKET}" \
  --set-env-vars="LIVEKIT_WORKER_MAX_LOAD=0.95"

echo ""
echo "✅ Deployment complete!"
echo "📋 View logs: gcloud run services logs read ${SERVICE_NAME} --region=${REGION}"
echo "📊 View service: gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
