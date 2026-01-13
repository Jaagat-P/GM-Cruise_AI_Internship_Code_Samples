// DOM Elements
let videoElement, questionInput, submitButton, resultDiv, videoFile;
let extractedSubtitles = '';

document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.width = 640;
    
    // Add subtitle track support
    const trackElement = document.createElement('track');
    trackElement.kind = 'subtitles';
    trackElement.label = 'English';
    trackElement.default = true;
    videoElement.appendChild(trackElement);

    // Setup video file input
    const videoInput = document.createElement('input');
    videoInput.type = 'file';
    videoInput.accept = 'video/*,.srt,.vtt';
    videoInput.className = 'video-input';
    
    // Setup question input
    questionInput = document.createElement('input');
    questionInput.type = 'text';
    questionInput.placeholder = 'Ask a question about the video...';
    questionInput.className = 'question-input';
    
    // Setup submit button
    submitButton = document.createElement('button');
    submitButton.textContent = 'Ask';
    submitButton.className = 'submit-button';
    
    // Setup result div
    resultDiv = document.createElement('div');
    resultDiv.className = 'result';
    
    // Add elements to page
    document.body.appendChild(videoInput);
    document.body.appendChild(videoElement);
    document.body.appendChild(questionInput);
    document.body.appendChild(submitButton);
    document.body.appendChild(resultDiv);

    // Handle video file selection
    videoInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        videoFile = file;
        videoElement.src = URL.createObjectURL(file);
        
        // If a subtitle file was uploaded
        if (file.name.endsWith('.srt') || file.name.endsWith('.vtt')) {
            trackElement.src = URL.createObjectURL(file);
        }
        
        // Listen for subtitle cues
        trackElement.addEventListener('cuechange', () => {
            const cues = trackElement.track.activeCues;
            if (cues && cues.length > 0) {
                for (const cue of cues) {
                    extractedSubtitles += cue.text + ' ';
                }
            }
        });
    });

    // Handle form submission
    submitButton.addEventListener('click', async () => {
        if (!videoFile || !questionInput.value.trim()) {
            alert('Please upload a video and enter a question');
            return;
        }

        submitButton.disabled = true;
        resultDiv.textContent = 'Processing...';

        try {
            // Extract video frames
            const frames = await extractFrames(videoElement);
            
            // Send frames, question, and subtitles to backend
            const response = await fetch('/api/gemini', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: questionInput.value,
                    videoFrames: frames,
                    subtitles: extractedSubtitles // Include extracted subtitles
                })
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            resultDiv.textContent = data.answer;
            
        } catch (error) {
            resultDiv.textContent = `Error: ${error.message}`;
        } finally {
            submitButton.disabled = false;
        }
    });
});

// Function to extract frames from video
async function extractFrames(videoElement) {
    const frames = [];
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Extract frames at 1-second intervals
    for (let i = 0; i < videoElement.duration; i += 1) {
        videoElement.currentTime = i;
        
        // Wait for the video to seek to the specified time
        await new Promise(resolve => {
            videoElement.addEventListener('seeked', resolve, { once: true });
        });
        
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0);
        
        // Convert frame to base64
        const base64Frame = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
        frames.push(base64Frame);
    }
    
    return frames;
}
