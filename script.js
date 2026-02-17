document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const ttsInput = document.getElementById('tts-input');
    const instructionInput = document.getElementById('instruction-input');
    const voiceSelector = document.getElementById('preset-voices');
    const audioPlayer = document.getElementById('audio-player');
    const outputCard = document.getElementById('output-card');
    const downloadBtn = document.getElementById('download-btn');
    const navItems = document.querySelectorAll('.nav-item');
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');

    // Tab Controls
    const presetVoicesDiv = document.getElementById('preset-voices');
    const cloneControls = document.getElementById('clone-controls');
    const designControls = document.getElementById('design-controls');
    const readerControls = document.getElementById('reader-controls');

    let currentTab = 'preset';
    let selectedVoice = null;
    let selectedFile = null;

    // Global Drag & Drop Prevention
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());

    // --- Streaming Queue Logic ---
    let playQueue = [];
    let currentSessionId = null;
    let currentlyPlayingIndex = -1;
    let isAutoPlaying = false;
    let totalChunksInSession = 0;
    let isPlaybackStalled = false;

    // Load voices
    async function loadVoices() {
        try {
            const res = await fetch('/api/voices');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const voices = await res.json();
            voiceSelector.innerHTML = voices.map(v => `
                <div class="voice-card ${(selectedVoice === v.id) ? 'selected' : ''}" data-id="${v.id}">
                    <div class="voice-badge">${v.type === 'preset' ? 'Official' : 'Custom'}</div>
                    ${v.type !== 'preset' ? `<button class="delete-voice-btn" title="Delete voice">×</button>` : ''}
                    <h4>${v.name}</h4>
                    <p>${v.description || ''}</p>
                </div>
            `).join('');

            document.querySelectorAll('.voice-card').forEach(card => {
                card.addEventListener('click', () => {
                    document.querySelectorAll('.voice-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');
                    selectedVoice = card.dataset.id;
                });
            });

            // Add delete listeners
            document.querySelectorAll('.delete-voice-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation(); // Avoid selecting the card
                    const card = e.target.closest('.voice-card');
                    const voiceId = card.dataset.id;
                    const voiceName = card.querySelector('h4').innerText;

                    if (confirm(`¿Estás seguro de que quieres eliminar la voz "${voiceName}"?`)) {
                        try {
                            const delRes = await fetch(`/api/voices/delete/${voiceId}`, { method: 'POST' });
                            if (delRes.ok) {
                                loadVoices(); // Refresh list
                            } else {
                                const err = await delRes.json();
                                alert('Error al eliminar: ' + (err.error || 'Unknown error'));
                            }
                        } catch (err) {
                            console.error('Error deleting voice', err);
                        }
                    }
                });
            });
        } catch (e) {
            console.error('Error loading voices', e);
        }
    }

    loadVoices();

    // Speed Slider Listener
    speedSlider.addEventListener('input', () => {
        speedValue.innerText = `${speedSlider.value}x`;
    });

    // Navigation
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            currentTab = item.dataset.tab;

            [presetVoicesDiv, cloneControls, designControls, readerControls].forEach(el => el.classList.add('hidden'));

            if (currentTab === 'preset') presetVoicesDiv.classList.remove('hidden');
            if (currentTab === 'clone') cloneControls.classList.remove('hidden');
            if (currentTab === 'design') designControls.classList.remove('hidden');
            if (currentTab === 'reader') readerControls.classList.remove('hidden');
        });
    });

    // File Upload handling
    const audioInput = document.getElementById('audio-input');
    const dropZone = document.getElementById('drop-zone');

    if (dropZone) {
        dropZone.addEventListener('click', () => audioInput.click());
        audioInput.addEventListener('change', (e) => {
            selectedFile = e.target.files[0];
            if (selectedFile) dropZone.querySelector('p').innerText = `Selected: ${selectedFile.name}`;
        });
    }

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

    async function saveVoice(type, value, audioFile = null) {
        const name = prompt(`Enter a name for this ${type} voice:`);
        if (!name) return;
        const description = prompt(`Enter a short description for "${name}":`);

        try {
            let res;
            if (type === 'clone' && audioFile) {
                const formData = new FormData();
                formData.append('name', name);
                formData.append('description', description || '');
                formData.append('type', type);
                formData.append('audio', audioFile);
                res = await fetch('/api/voices/save', { method: 'POST', body: formData });
            } else {
                res = await fetch('/api/voices/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, description, type, value })
                });
            }

            const result = await res.json();
            if (result.success) {
                alert('¡Voz guardada correctamente!');
                loadVoices();
            } else {
                alert('Error al guardar: ' + result.error);
            }
        } catch (e) {
            console.error(e);
            alert('Error de conexión al guardar la voz');
        }
    }

    saveDesignBtn.addEventListener('click', () => {
        const desc = document.getElementById('voice-description').value;
        if (!desc) return alert('Enter a description first');
        saveVoice('design', desc);
    });

    saveCloneBtn.addEventListener('click', () => {
        if (!selectedFile) return alert('Por favor, selecciona o arrastra un archivo de audio primero');
        saveVoice('clone', null, selectedFile);
    });

    // --- UI Progress Elements ---
    const progressContainer = document.getElementById('progress-container');
    const progressStatus = document.getElementById('progress-status');
    const progressCount = document.getElementById('progress-count');
    const progressBarFill = document.getElementById('progress-bar-fill');

    // --- Unified Generation & Polling ---
    generateBtn.addEventListener('click', async () => {
        const text = ttsInput.value.trim();
        const language = document.getElementById('language-input').value;
        if (!text) return alert('Please enter some text');

        // Reset UI
        generateBtn.disabled = true;
        generateBtn.innerText = 'Initializing...';
        progressContainer.classList.remove('hidden');
        progressBarFill.style.width = '0%';
        progressStatus.innerText = 'Preparing...';
        progressCount.innerText = '0/0';
        outputCard.classList.add('hidden');

        try {
            // Check Health first
            const healthRes = await fetch('/api/health');
            if (healthRes.ok) {
                const health = await healthRes.json();
                if (!health.ready) {
                    if (health.error) throw new Error('Model loading failed: ' + health.error);
                    progressStatus.innerText = 'Downloading/Loading Models (First time may take 5-10 mins)...';
                    setTimeout(() => generateBtn.click(), 5000);
                    return;
                }
            }

            // Reset Streaming State
            playQueue = [];
            currentlyPlayingIndex = -1;
            isAutoPlaying = false;
            currentSessionId = null;
            isPlaybackStalled = false;

            // 1. Start Session
            let payload = {
                text,
                language,
                speed: parseFloat(speedSlider.value),
                voice_id: selectedVoice || 'vivian',
                engine: document.getElementById('engine-input').value,
                mode: currentTab === 'reader' ? 'preset' : currentTab,
                instruction: instructionInput.value
            };

            if (currentTab === 'design') {
                payload.extra_info = document.getElementById('voice-description').value;
                if (!payload.extra_info) throw new Error('Enter voice description');
            }

            // Execute based on type
            if (currentTab === 'clone' && selectedFile) {
                const formData = new FormData();
                formData.append('audio', selectedFile);
                formData.append('text', text);
                formData.append('language', language);
                formData.append('speed', speedSlider.value);

                const res = await fetch('/api/stream/start-clone', { method: 'POST', body: formData });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `Server error (${res.status})`);
                }
                const session = await res.json();
                if (session.error) throw new Error(session.error);

                currentSessionId = session.session_id;
                totalChunksInSession = session.total_chunks;
                pollGenerationStatus(session.session_id, session.total_chunks);
                return;
            }

            const res = await fetch('/api/stream/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || `Server error (${res.status})`);
            }
            const session = await res.json();
            if (session.error) throw new Error(session.error);

            currentSessionId = session.session_id;
            totalChunksInSession = session.total_chunks;
            pollGenerationStatus(session.session_id, session.total_chunks);

        } catch (e) {
            alert(e.message);
            resetGenUI();
        }
    });

    async function pollGenerationStatus(sessionId, totalChunks) {
        if (currentSessionId !== sessionId) return;

        try {
            const res = await fetch(`/api/stream/status/${sessionId}`);
            if (!res.ok) throw new Error(`Status lost (${res.status})`);
            const status = await res.json();

            if (status.error) throw new Error(status.error);

            // Update Progress UI
            const readyCount = status.ready_chunks.length;
            const progress = (readyCount / totalChunks) * 100;
            progressBarFill.style.width = `${progress}%`;
            progressCount.innerText = `${readyCount}/${totalChunks}`;

            // Update Status Text with Context
            let statusText = status.status;
            if (isPlaybackStalled) {
                statusText = "Recuperando buffer fluidez...";
            } else if (!isAutoPlaying && status.ready_chunks.length > 0) {
                statusText = `Cargando buffer de seguridad...`;
            }
            progressStatus.innerText = statusText;

            // Update Play Queue
            status.ready_chunks.forEach(idx => {
                if (!playQueue.includes(idx)) {
                    playQueue.push(idx);
                    playQueue.sort((a, b) => a - b);
                }
            });

            // Auto-start playback
            if (!isAutoPlaying && playQueue.includes(0)) {
                isAutoPlaying = true;
                outputCard.classList.remove('hidden');
                playNextChunk();
            }

            // Stall Recovery
            if (isPlaybackStalled && playQueue.includes(currentlyPlayingIndex + 1)) {
                isPlaybackStalled = false;
                playNextChunk();
            }

            if (status.status === 'Completed' && readyCount === totalChunks) {
                console.log('All chunks ready.');
            } else if (status.status === 'Error') {
                throw new Error(status.error);
            } else {
                setTimeout(() => pollGenerationStatus(sessionId, totalChunks), 1500);
            }
        } catch (e) {
            console.error('Generation error:', e);
            resetGenUI();
        }
    }

    function playNextChunk() {
        const nextIndex = currentlyPlayingIndex + 1;
        if (playQueue.includes(nextIndex)) {
            isPlaybackStalled = false;
            currentlyPlayingIndex = nextIndex;
            audioPlayer.src = `/api/stream/audio/${currentSessionId}/${nextIndex}`;
            audioPlayer.play().catch(e => console.warn("Auto-play blocked or error:", e));
        } else {
            console.log("Buffering...");
            isPlaybackStalled = true;
            progressStatus.innerText = "Buffering next chunk...";
        }
    }

    audioPlayer.addEventListener('ended', () => {
        if (currentlyPlayingIndex < totalChunksInSession - 1) {
            playNextChunk();
        } else {
            finalizeSession();
        }
    });

    async function finalizeSession() {
        progressStatus.innerText = 'Finishing...';
        // Check if there is a concatenate endpoint, otherwise just reset
        try {
            const concatRes = await fetch(`/api/stream/concatenate/${currentSessionId}`);
            if (concatRes.ok) {
                const final = await concatRes.json();
                if (final.url) {
                    downloadBtn.onclick = () => window.open(final.url);
                }
            }
        } catch (e) {
            console.log("Session finished (no concatenation needed/available)");
        }
        resetGenUI();
    }

    function resetGenUI() {
        generateBtn.disabled = false;
        generateBtn.innerText = 'Generate Voice';
        progressContainer.classList.add('hidden');
    }

    // --- PDF Extraction Utility ---
    const pdfInput = document.getElementById('pdf-input');
    const pdfDropZone = document.getElementById('pdf-drop-zone');

    if (pdfDropZone) {
        pdfDropZone.addEventListener('click', () => pdfInput.click());

        pdfDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            pdfDropZone.style.borderColor = 'var(--primary)';
        });
        pdfDropZone.addEventListener('dragleave', () => {
            pdfDropZone.style.borderColor = 'var(--glass-border)';
        });
        pdfDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            pdfDropZone.style.borderColor = 'var(--glass-border)';
            const file = e.dataTransfer.files[0];
            if (file && file.type === 'application/pdf') {
                handlePDFExtraction(file);
            }
        });
    }

    pdfInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) handlePDFExtraction(file);
    });

    async function handlePDFExtraction(file) {
        pdfDropZone.querySelector('p').innerText = 'Extracting text...';
        const formData = new FormData();
        formData.append('pdf', file);

        try {
            const res = await fetch('/api/extract-pdf', { method: 'POST', body: formData });
            if (!res.ok) throw new Error(`Server error (${res.status})`);
            const result = await res.json();
            if (result.text) {
                ttsInput.value = result.text.trim();
                pdfDropZone.querySelector('p').innerText = `Extracted: ${file.name}`;
            } else {
                alert('Extraction failed');
                pdfDropZone.querySelector('p').innerText = 'Drop PDF to extract text';
            }
        } catch (e) {
            console.error(e);
            alert("Error al extraer PDF: " + e.message);
            pdfDropZone.querySelector('p').innerText = 'Drop PDF to extract text';
        }
    }
});
