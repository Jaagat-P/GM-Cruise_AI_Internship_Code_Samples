const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

// Add global fetch for Node.js
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Gemini API endpoint
app.post('/api/gemini', async (req, res) => {
    try {
        const { question, videoFrames, captions } = req.body;
        
        if (!process.env.GEMINI_API_KEY) {
            return res.status(500).json({ 
                error: 'Gemini API key not configured. Please add GEMINI_API_KEY to your .env file' 
            });
        }

        // Build the request for Gemini API
        const parts = [
            {
                text: `You are analyzing a video about automobile production automation assembly lines. The user has uploaded a video and is asking questions about it. Here's their question: "${question}".`
            }
        ];

        // Add captions context if available
        if (captions && captions.trim()) {
            parts.push({
                text: `Here are the video captions with timestamps:\n${captions.trim()}`
            });
        }

        parts.push({
            text: "Please provide a detailed analysis based on both the video frames and the captions. Reference specific timestamps when relevant to your answer."
        });

        // Add video frames to the request
        if (videoFrames && videoFrames.length > 0) {
            videoFrames.forEach(frame => {
                parts.push({
                    inline_data: {
                        mime_type: "image/jpeg",
                        data: frame
                    }
                });
            });
        }

        const requestBody = {
            contents: [{
                parts: parts
            }],
            generationConfig: {
                temperature: 0.7,
                maxOutputTokens: 2048,
            }
        };

        // Make request to Gemini API
        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            }
        );

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = 'Failed to get response from Gemini';
            
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.error?.message || errorMessage;
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
            
            return res.status(response.status).json({
                error: `Gemini API Error: ${errorMessage}`
            });
        }

        const data = await response.json();
        
        if (!data.candidates || !data.candidates[0] || !data.candidates[0].content || !data.candidates[0].content.parts) {
            return res.status(500).json({
                error: 'Invalid response format from Gemini API'
            });
        }
        
        const answer = data.candidates[0].content.parts[0].text;
        res.json({ answer });
        
    } catch (error) {
        console.error('Server Error:', error);
        res.status(500).json({ 
            error: `Server error: ${error.message}` 
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        timestamp: new Date().toISOString(),
        hasApiKey: !!process.env.GEMINI_API_KEY
    });
});

// Serve the main HTML file for all other routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:3000`);
    console.log(`📁 Serving files from: ${path.join(__dirname, 'public')}`);
    console.log(`🔑 API Key configured: ${!!process.env.GEMINI_API_KEY}`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
    console.log('👋 Server shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('👋 Server shutting down gracefully...');
    process.exit(0);
});
