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
            // 1. Start Session
            let payload = {
                text,
                language,
                speed: parseFloat(speedSlider.value),
                voice_id: selectedVoice || 'vivian',
                mode: currentTab,
                instruction: instructionInput.value
            };

            // Additional info for specialized modes
            if (currentTab === 'design') {
                payload.extra_info = document.getElementById('voice-description').value;
                if (!payload.extra_info) throw new Error('Enter voice description');
            } else if (currentTab === 'clone') {
                if (!selectedFile) throw new Error('Please upload a reference audio');
                // For direct clones not yet saved, we still need to handle the initial upload
                // However, following the plan to unify, we keep it simple: 
                // if it's a new clone, we use the old direct endpoint or a simplified session start.
                // TO-DO: For now, we'll keep the direct /api/clone logic for unsaved clones 
                // but wrap it in UI feedback, OR simplify sessions to handle files (harder).
                // Let's stick to the simplest: Sessions for everything else, direct for unsaved clone.
            }

            // Execute based on type
            if (currentTab === 'clone' && selectedFile) {
                // Direct clone path (simplified progress)
                progressStatus.innerText = 'Cloning... (Slow process)';
                progressBarFill.style.width = '50%';

                const formData = new FormData();
                formData.append('audio', selectedFile);
                formData.append('text', text);
                formData.append('language', language);
                formData.append('speed', speedSlider.value);

                const res = await fetch('/api/clone', { method: 'POST', body: formData });
                const result = await res.json();
                if (result.url) finalizeGeneration(result.url);
                else throw new Error(result.error);
                return;
            }

            // Session Path (Presets & Design)
            const res = await fetch('/api/stream/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const session = await res.json();

            if (session.error) throw new Error(session.error);
            pollGenerationStatus(session.session_id, session.total_chunks);

        } catch (e) {
            alert(e.message);
            resetGenUI();
        }
    });

    async function pollGenerationStatus(sessionId, totalChunks) {
        try {
            const res = await fetch(`/api/stream/status/${sessionId}`);
            const status = await res.json();

            if (status.error) throw new Error(status.error);

            // Update UI
            const ready = status.ready_chunks.length;
            const progress = (ready / totalChunks) * 100;
            progressBarFill.style.width = `${progress}%`;
            progressCount.innerText = `${ready}/${totalChunks}`;
            progressStatus.innerText = status.status;

            if (status.status === 'Completed') {
                progressStatus.innerText = 'Merging chunks...';
                const concatRes = await fetch(`/api/stream/concatenate/${sessionId}`);
                const final = await concatRes.json();
                if (final.url) finalizeGeneration(final.url);
                else throw new Error(final.error);
            } else if (status.status === 'Error') {
                throw new Error(status.error);
            } else {
                // Continue polling
                setTimeout(() => pollGenerationStatus(sessionId, totalChunks), 1500);
            }
        } catch (e) {
            alert('Generation error: ' + e.message);
            resetGenUI();
        }
    }

    function finalizeGeneration(url) {
        audioPlayer.src = url;
        outputCard.classList.remove('hidden');
        audioPlayer.play();
        downloadBtn.onclick = () => window.open(url);
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
                pdfDropZone.querySelector('p').innerText = 'Drop PDF to extract text';
            }
        } catch (e) {
            console.error(e);
            pdfDropZone.querySelector('p').innerText = 'Drop PDF to extract text';
        }
    });
});
