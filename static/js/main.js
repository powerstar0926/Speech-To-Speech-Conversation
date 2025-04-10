// Get the current port from the URL
const port = window.location.port;
const socket = io('http://' + window.location.hostname + ':' + port);

const messagesContainer = document.getElementById('messages');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');

function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    messageDiv.textContent = text;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageDiv;
}

startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_recording', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        if (data.status === 'recording started') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.textContent = 'Recording...';
            statusDiv.className = 'status recording';
        }
    } catch (error) {
        console.error('Error starting recording:', error);
        statusDiv.textContent = 'Error starting recording. Please try again.';
    }
});

stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/stop_recording', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        if (data.status === 'recording stopped') {
            stopBtn.disabled = true;
            startBtn.disabled = false;
            statusDiv.textContent = 'Not Recording';
            statusDiv.className = 'status not-recording';
        }
    } catch (error) {
        console.error('Error stopping recording:', error);
        statusDiv.textContent = 'Error stopping recording. Please try again.';
    }
});

socket.on('connect', () => {
    console.log('Connected to server');
    statusDiv.textContent = 'Connected to server';
    setTimeout(() => {
        statusDiv.textContent = 'Not Recording';
        statusDiv.className = 'status not-recording';
    }, 2000);
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusDiv.textContent = 'Disconnected from server';
    startBtn.disabled = true;
    stopBtn.disabled = true;
});

socket.on('user_message', (data) => {
    addMessage(data.text, true);
});

// Handle streaming AI messages
const aiMessages = new Map();

socket.on('ai_message_start', (data) => {
    const messageDiv = document.createElement('div');
    messageDiv.id = data.id;
    messageDiv.className = 'message ai-message';
    messagesContainer.appendChild(messageDiv);
    aiMessages.set(data.id, messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
});

socket.on('ai_message_chunk', (data) => {
    const messageDiv = aiMessages.get(data.id);
    if (messageDiv) {
        messageDiv.textContent += data.text;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});

socket.on('ai_message_complete', (data) => {
    aiMessages.delete(data.id);
});

socket.on('interrupted', (data) => {
    statusDiv.textContent = data.message;
    setTimeout(() => {
        statusDiv.textContent = 'Recording...';
        statusDiv.className = 'status recording';
    }, 2000);
});

socket.on('error', (data) => {
    statusDiv.textContent = `Error: ${data.message}`;
    statusDiv.className = 'status not-recording';
    stopBtn.disabled = true;
    startBtn.disabled = false;
}); 