require('dotenv').config();
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const { AgentDispatchClient } = require('livekit-server-sdk');

const url = process.env.LIVEKIT_URL;
const apiKey = process.env.LIVEKIT_API_KEY;
const apiSecret = process.env.LIVEKIT_API_SECRET;

if (!url || !apiKey || !apiSecret) {
    console.error("Missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET in environment");
    console.error("Please ensure .env contains these variables.");
    process.exit(1);
}

const dispatchClient = new AgentDispatchClient(url, apiKey, apiSecret);

const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

app.post('/call', async (req, res) => {
    const { phone_number } = req.body;

    if (!phone_number) {
        return res.status(400).json({ error: "Missing 'phone_number' in request body." });
    }

    const roomName = `outbound-${Math.floor(Math.random() * 10000000000).toString()}`;
    console.log(`Dispatching agent 'dummyCode' to room '${roomName}' for outbound call to ${phone_number}...`);

    try {
        const dispatch = await dispatchClient.createDispatch(roomName, "dummyCode", {
            metadata: JSON.stringify({ phone_number })
        });

        console.log("Successfully created agent dispatch:", dispatch.id);
        return res.status(200).json({
            success: true,
            message: "Call dispatched successfully",
            dispatchId: dispatch.id,
            roomName: roomName
        });
    } catch (err) {
        console.error("Error creating dispatch:", err);
        return res.status(500).json({ error: "Failed to dispatch call", details: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Outbound caller API server running on http://localhost:${PORT}`);
    console.log(`Send a POST request to http://localhost:${PORT}/call with JSON body { "phone_number": "+1234567890" } to trigger a call.`);
});
