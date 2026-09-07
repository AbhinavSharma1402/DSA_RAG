// YouTube DSA RAG Tutor - Frontend JavaScript

class YouTubeDSARagTutor {
    constructor() {
        this.conversationId = null;
        this.selectedVideoId = null;
        this.selectedTimestamp = null;
        this.selectedSource = null;
        this.isProcessing = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.createNewConversation();
    }

    initializeElements() {
        // Input and buttons
        this.queryInput = document.getElementById('query-input');
        this.askBtn = document.getElementById('ask-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');
        
        // Filters
        this.topicFilter = document.getElementById('topic-filter');
        this.modeSelect = document.getElementById('mode-select');
        
        // Display areas
        this.conversationHistory = document.getElementById('conversation-history');
        this.answerContainer = document.getElementById('answer-container');
        this.youtubePlayer = document.getElementById('youtube-player');
        this.playerTimestamp = document.getElementById('player-timestamp');
        
        // Modals
        this.loadingIndicator = document.getElementById('loading-indicator');
        this.errorModal = document.getElementById('error-modal');
        this.errorMessage = document.getElementById('error-message');
    }

    attachEventListeners() {
        // Input and buttons
        this.askBtn.addEventListener('click', () => this.handleAsk());
        this.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleAsk();
        });
        this.newChatBtn.addEventListener('click', () => this.handleNewChat());
        
        // Filters
        this.topicFilter.addEventListener('change', () => {
            console.log('Topic filter changed to:', this.topicFilter.value);
        });
        this.modeSelect.addEventListener('change', () => {
            console.log('Mode changed to:', this.modeSelect.value);
        });
    }

    async createNewConversation() {
        try {
            // Generate a new conversation ID locally
            this.conversationId = this.generateUUID();
            console.log('Created conversation:', this.conversationId);
            
            // Clear UI
            this.conversationHistory.innerHTML = `
                <div class="welcome-message">
                    <h2>Welcome! 👋</h2>
                    <p>Ask a question about DSA lectures and get answers grounded in actual lecture content.</p>
                    <p><strong>Examples:</strong></p>
                    <ul>
                        <li>"What is memoization?"</li>
                        <li>"Difference between memoization and tabulation"</li>
                        <li>"How does DFS work?"</li>
                    </ul>
                </div>
            `;
            this.answerContainer.innerHTML = `
                <div class="placeholder">
                    <p>Answers will appear here with citations and timestamps</p>
                </div>
            `;
            this.youtubePlayer.src = '';
            this.playerTimestamp.textContent = 'Ready to play a lecture';
            
        } catch (error) {
            console.error('Error creating conversation:', error);
            this.showError('Failed to create new conversation');
        }
    }

    async handleAsk() {
        const query = this.queryInput.value.trim();
        
        if (!query) {
            this.showError('Please enter a question');
            return;
        }
        
        if (query.length > 2000) {
            this.showError('Question is too long (max 2000 characters)');
            return;
        }
        
        if (this.isProcessing) {
            return;
        }
        
        // Add user message to history
        this.addMessageToHistory(query, 'user');
        this.queryInput.value = '';
        
        // Show loading
        this.setProcessing(true);
        
        try {
            const response = await this.callAPI('/api/chat', {
                query,
                conversation_id: this.conversationId,
                topic: this.topicFilter.value,
                mode: this.modeSelect.value
            });
            
            // Display answer
            this.displayAnswer(response);
            
            // Add assistant message to history
            this.addMessageToHistory(response.answer, 'assistant');
            
        } catch (error) {
            console.error('Error processing query:', error);
            this.showError('Failed to process your question. Please try again.');
        } finally {
            this.setProcessing(false);
        }
    }

    async displayAnswer(response) {
        const {
            answer,
            confidence,
            rewritten_query,
            sources,
            retrieved_chunks
        } = response;
        
        // Build HTML
        let html = '';
        
        // Confidence badge
        html += `<span class="confidence-badge ${confidence}">Confidence: ${confidence.toUpperCase()}</span>`;
        html += '\n';
        
        // Answer text
        html += `<div class="answer-text">${this.escapeHtml(answer)}</div>`;
        
        // Rewritten query info
        if (rewritten_query) {
            html += `<p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 15px;">`;
            html += `<strong>Query Rewritten:</strong> "${this.escapeHtml(rewritten_query)}"</p>`;
        }
        
        // Sources section
        if (sources && sources.length > 0) {
            html += `<div class="sources-section">`;
            html += `<h4>📹 Sources (Click to jump to lecture)</h4>`;
            
            sources.forEach((source, index) => {
                html += `
                    <div class="source-item" onclick="app.selectSource(${index}, '${source.video_id}', ${source.timestamp})">
                        <div class="source-title">
                            <span class="source-icon">[${index + 1}]</span>
                            ${this.escapeHtml(source.video_title)}
                        </div>
                        <div class="source-timestamp">${source.timestamp_display}</div>
                    </div>
                `;
            });
            
            html += `</div>`;
        } else {
            html += `<p style="color: var(--text-secondary); font-size: 13px; margin-top: 15px;">`;
            html += `No sources found in lecture material.</p>`;
        }
        
        // Update container
        this.answerContainer.innerHTML = html;
        
        // Store sources for player interaction
        this.currentSources = sources || [];
    }

    selectSource(index, videoId, timestamp) {
        // Update visual selection
        const items = document.querySelectorAll('.source-item');
        items.forEach((item, i) => {
            item.classList.toggle('active', i === index);
        });
        
        this.selectedVideoId = videoId;
        this.selectedTimestamp = timestamp;
        this.selectedSource = index;
        
        // Load YouTube player
        this.loadYoutubePlayer(videoId, timestamp);
    }

    loadYoutubePlayer(videoId, timestamp) {
        // YouTube embed URL with timestamp
        const url = `https://www.youtube.com/embed/${videoId}?start=${timestamp}&autoplay=1`;
        this.youtubePlayer.src = url;
        
        // Update timestamp display
        const timeDisplay = this.formatTimestamp(timestamp);
        this.playerTimestamp.textContent = `Currently viewing: ${timeDisplay}`;
    }

    addMessageToHistory(content, role) {
        const messageHtml = `
            <div class="message ${role}">
                <div class="message-content">${this.escapeHtml(content)}</div>
            </div>
        `;
        
        this.conversationHistory.innerHTML += messageHtml;
        
        // Scroll to bottom
        this.conversationHistory.scrollTop = this.conversationHistory.scrollHeight;
    }

    async callAPI(endpoint, data) {
        const url = `http://localhost:8000${endpoint}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API error: ${response.status}`);
        }
        
        return await response.json();
    }

    async handleNewChat() {
        const confirmed = confirm('Start a new conversation? Current chat will be cleared.');
        if (confirmed) {
            await this.createNewConversation();
        }
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.askBtn.disabled = processing;
        this.queryInput.disabled = processing;
        
        if (processing) {
            this.loadingIndicator.classList.remove('hidden');
        } else {
            this.loadingIndicator.classList.add('hidden');
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.remove('hidden');
    }

    formatTimestamp(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
}

// Global helper functions
function closeErrorModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

// Initialize when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new YouTubeDSARagTutor();
    console.log('YouTube DSA RAG Tutor initialized');
});
