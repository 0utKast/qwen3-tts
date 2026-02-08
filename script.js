document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const ttsInput = document.getElementById('tts-input');
    const instructionInput = document.getElementById('instruction-input');
    const voiceSelector = document.getElementById('preset-voices');
    const audioPlayer = document.getElementById('audio-player');
    const outputCard = document.getElementById('output-card');
    const downloadBtn = document.getElementById('download-btn');
    const navItems = document.querySelectorAll('.nav-item');

    // Tab Controls
    const presetVoicesDiv = document.getElementById('preset-voices');
    const cloneControls = document.getElementById('clone-controls');
    const designControls = document.getElementById('design-controls');

    let currentTab = 'preset';
    let selectedVoice = null;
    let selectedFile = null;

    // Load voices
    async function loadVoices() {
        try {
            const res = await fetch('/api/voices');
            const voices = await res.json();
            voiceSelector.innerHTML = voices.map(v => `
                <div class="voice-card ${v.type || 'preset'}" data-id="${v.id}">
                    <div class="voice-badge">${v.type === 'preset' ? 'Official' : 'Custom'}</div>
                    <h4>${v.name}</h4>
                    <p>${v.description}</p>
                </div>
            `).join('');

            document.querySelectorAll('.voice-card').forEach(card => {
                card.addEventListener('click', () => {
                    document.querySelectorAll('.voice-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');
                    selectedVoice = card.dataset.id;
                });
            });
        } catch (e) {
            console.error('Error loading voices', e);
        }
    }

    loadVoices();

    // Navigation
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            currentTab = item.dataset.tab;

            presetVoicesDiv.classList.add('hidden');
            cloneControls.classList.add('hidden');
            designControls.classList.add('hidden');

            if (currentTab === 'preset') {
                presetVoicesDiv.classList.remove('hidden');
                sessionMode = 'preset';
            }
            if (currentTab === 'clone') {
                cloneControls.classList.remove('hidden');
                sessionMode = 'clone';
            }
            if (currentTab === 'design') {
                designControls.classList.remove('hidden');
                sessionMode = 'design';
            }
            if (currentTab === 'reader') document.getElementById('reader-controls').classList.remove('hidden');
        });
    });

    // File Upload handling
    const dropZone = document.getElementById('drop-zone');
    const audioInput = document.getElementById('audio-input');

    dropZone.addEventListener('click', () => audioInput.click());
    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            dropZone.querySelector('p').innerText = `Selected: ${selectedFile.name}`;
        }
    });

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--glass-border)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--glass-border)';
        const file = e.dataTransfer.files[0];
        if (file) {
            selectedFile = file;
            dropZone.querySelector('p').innerText = `Selected: ${selectedFile.name}`;
            audioInput.files = e.dataTransfer.files; // Sync to input
        }
    });
    // Save Voice Logic
    const saveDesignBtn = document.getElementById('save-design-btn');
    const saveCloneBtn = document.getElementById('save-clone-btn');

    async function saveVoice(type, value) {
        const name = prompt(`Enter a name for this ${type} voice:`);
        if (!name) return;

        const description = prompt(`Enter a short description for "${name}":`);

        try {
            const res = await fetch('/api/voices/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description, type, value })
            });
            const result = await res.json();
            if (result.success) {
                alert('Voice saved to library!');
                loadVoices();
            } else {
                alert('Error saving: ' + result.error);
            }
        } catch (e) {
            console.error(e);
        }
    }

    saveDesignBtn.addEventListener('click', () => {
        const desc = document.getElementById('voice-description').value;
        if (!desc) return alert('Enter a description first');
        saveVoice('design', desc);
    });

    saveCloneBtn.addEventListener('click', () => {
        if (!selectedFile) return alert('Please select or drop an audio file first');
        // Since the file is handled via FormData in /api/clone, for simplification 
        // we'll "save" after a successful clone or by tagging the last uploaded file.
        // For now, we'll assume the backend can find 'temp_ref.wav' or we pass the context.
        saveVoice('clone', 'temp_ref.wav');
    });
    // Generation
    generateBtn.addEventListener('click', async () => {
        const text = ttsInput.value.trim();
        const language = document.getElementById('language-input').value;
        if (!text) return alert('Please enter some text');

        generateBtn.disabled = true;
        generateBtn.innerText = 'Generating...';

        try {
            let endpoint = '/api/generate';
            let body;
            let options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            };

            if (currentTab === 'preset') {
                body = JSON.stringify({
                    text,
                    voice_id: selectedVoice || 'vivian',
                    instruction: instructionInput.value,
                    language
                });
            } else if (currentTab === 'clone') {
                if (!selectedFile) return alert('Please upload a reference audio');
                endpoint = '/api/clone';
                const formData = new FormData();
                formData.append('audio', selectedFile);
                formData.append('text', text);
                formData.append('language', language);
                options = { method: 'POST', body: formData };
            } else if (currentTab === 'design') {
                const desc = document.getElementById('voice-description').value;
                if (!desc) return alert('Enter voice description');
                endpoint = '/api/design';
                body = JSON.stringify({
                    text,
                    description: desc,
                    language
                });
            }

            if (body) options.body = body;

            const res = await fetch(endpoint, options);
            const result = await res.json();

            if (result.url) {
                audioPlayer.src = result.url;
                outputCard.classList.remove('hidden');
                streamCard.classList.add('hidden'); // Hide streaming if doing single gen
                audioPlayer.play();
                downloadBtn.onclick = () => window.open(result.url);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (e) {
            alert('Generation failed. Check backend logs.');
        } finally {
            generateBtn.disabled = false;
            generateBtn.innerText = 'Generate Voice';
        }
    });

    // --- Streaming & Reader Logic ---
    const pdfInput = document.getElementById('pdf-input');
    const pdfDropZone = document.getElementById('pdf-drop-zone');
    const streamCard = document.getElementById('stream-card');
    const streamInfo = document.getElementById('stream-info');
    const streamProgress = document.getElementById('stream-progress-bar');
    const streamCount = document.getElementById('stream-count');
    const stopStreamBtn = document.getElementById('stop-stream-btn');
    const player1 = document.getElementById('player-1');
    const player2 = document.getElementById('player-2');

    let currentSession = null;
    let playingIndex = -1;
    let readyChunks = [];
    let isStreaming = false;
    let activePlayer = 1;
    let sessionMode = 'preset';

    pdfDropZone.addEventListener('click', () => pdfInput.click());
    pdfInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        pdfDropZone.querySelector('p').innerText = 'Extracting text...';
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/extract-text', { method: 'POST', body: formData });
            const result = await res.json();
            if (result.text) {
                ttsInput.value = result.text;
                pdfDropZone.querySelector('p').innerText = `Extracted: ${file.name}`;
            } else {
                alert('Extraction failed: ' + result.error);
            }
        } catch (e) {
            console.error(e);
        }
    });

    async function startStreaming() {
        const text = ttsInput.value.trim();
        const language = document.getElementById('language-input').value;
        const designPrompt = document.getElementById('voice-description').value;
        if (!text) return alert('Please enter or extract text first');

        generateBtn.disabled = true;
        generateBtn.innerText = 'Initializing...';

        try {
            const res = await fetch('/api/stream/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    voice_id: selectedVoice || 'vivian',
                    instruction: instructionInput.value,
                    language,
                    mode: sessionMode,
                    extra_info: sessionMode === 'design' ? designPrompt : ''
                })
            });

            currentSession = await res.json();
            if (currentSession.session_id) {
                isStreaming = true;
                playingIndex = -1;
                readyChunks = [];
                activePlayer = 1;

                outputCard.classList.add('hidden');
                streamCard.classList.remove('hidden');

                // Reset UI
                streamProgress.style.width = '0%';
                streamCount.innerText = '0/0';
                streamInfo.innerText = 'Starting...';

                pollStatus();
            }
        } catch (e) {
            alert('Streaming failed to start');
            generateBtn.disabled = false;
            generateBtn.innerText = 'Generate Voice';
        }
    }

    async function pollStatus() {
        if (!isStreaming) return;

        try {
            const res = await fetch(`/api/stream/status/${currentSession.session_id}`);
            const status = await res.json();
            readyChunks = status.ready_chunks;

            streamInfo.innerText = status.status;
            const progress = (readyChunks.length / currentSession.total_chunks) * 100;
            streamProgress.style.width = `${progress}%`;
            streamCount.innerText = `${readyChunks.length}/${currentSession.total_chunks}`;

            // Auto-start playback when first chunk is ready
            if (playingIndex === -1 && readyChunks.includes(0)) {
                playNextChunk();
            }

            if (status.status !== 'Completed' && status.status !== 'Error' && isStreaming) {
                setTimeout(pollStatus, 2000);
            }
        } catch (e) {
            console.error('Status poll error', e);
        }
    }

    function playNextChunk() {
        if (!isStreaming) return;

        const nextIndex = playingIndex + 1;
        if (nextIndex >= currentSession.total_chunks) {
            isStreaming = false;
            streamInfo.innerText = 'Finished Reading';
            generateBtn.disabled = false;
            generateBtn.innerText = 'Generate Voice';
            return;
        }

        if (readyChunks.includes(nextIndex)) {
            playingIndex = nextIndex;
            const audioSrc = `/api/stream/audio/${currentSession.session_id}/${playingIndex}`;

            const currentPlayer = activePlayer === 1 ? player1 : player2;
            const otherPlayer = activePlayer === 1 ? player2 : player1;

            currentPlayer.src = audioSrc;
            currentPlayer.play();

            // Preload next
            if (readyChunks.includes(playingIndex + 1)) {
                otherPlayer.src = `/api/stream/audio/${currentSession.session_id}/${playingIndex + 1}`;
                otherPlayer.load();
            }

            activePlayer = activePlayer === 1 ? 2 : 1;
        } else {
            streamInfo.innerText = 'Waiting for buffer...';
            setTimeout(playNextChunk, 1000);
        }
    }

    [player1, player2].forEach(p => {
        p.onended = () => {
            playNextChunk();
        };
        p.onerror = () => {
            console.error('Playback error on chunk', playingIndex);
            setTimeout(playNextChunk, 2000);
        };
    });

    stopStreamBtn.addEventListener('click', () => {
        isStreaming = false;
        player1.pause();
        player2.pause();
        streamInfo.innerText = 'Stopped';
        generateBtn.disabled = false;
        generateBtn.innerText = 'Generate Voice';
    });

    // Override Generate button if in reader mode
    generateBtn.addEventListener('click', (e) => {
        if (currentTab === 'reader') {
            e.stopImmediatePropagation();
            startStreaming();
        }
    }, true);
});
