# Implementation Plan - Progress Indicators & Cleanup

Add real-time progress feedback both in the terminal and in the web interface, while removing the non-viable Doc Reader streaming functionality.

## User Review Required

> [!IMPORTANT]
> - **Unified Generation**: "Standard Mode" generations (Presets, Clone, Design) will now use the session-based API internally. This allows showing a real-time progress bar.
> - **Doc Reader Removal**: The "Streaming Reader" UI and dual-player logic will be removed as it is slow and non-viable. The PDF extraction feature will be kept as a text-utility that fills the main editor.

## Proposed Changes

### Backend (`app.py`)

#### [MODIFY] [app.py](file:///c:/proyectos_python/Qwen3-TTS/app.py)
- **Terminal Feedback**: Add explicit `print` statements in `generation_worker` to show progress in the console (e.g., `Chunk 3/10 completed`).
- **Concatenation Endpoint [NEW]**: Add `/api/stream/concatenate/<session_id>` to merge all generated chunks into a single final WAV file.
- **Worker Logic**: Ensure all generation modes (preset, clone, design) use the `generation_worker` to support progress tracking.

### Frontend (`index.html` & `style.css`)

#### [MODIFY] [index.html](file:///c:/proyectos_python/Qwen3-TTS/index.html)
- **Progress UI**: Add a progress bar container below the main text editor.
- **Cleanup**: Remove the `stream-card` section and the dual audio players (`player-1`, `player-2`).
- **Reader Tab**: Rename "Doc Reader" to "PDF Extract" and simplify its control to just the upload box.

#### [MODIFY] [style.css](file:///c:/proyectos_python/Qwen3-TTS/style.css)
- Style the new unified progress bar with modern animations.

### Frontend Logic (`script.js`)

#### [MODIFY] [script.js](file:///c:/proyectos_python/Qwen3-TTS/script.js)
- **Polling System**: Rewrite the polling logic to be used by the main `generateBtn`.
- **Merge Chunks**: After polling reaches 100%, call the concatenate endpoint and load the final file into the player.
- **Remove Streaming Logic**: Eliminate `startStreaming`, `playNextChunk`, and the dual player event listeners.

---

## Verification Plan

### Manual Verification
1.  **Terminal Check**: Verify that the console prints progress updates during generation.
2.  **UI Bar**: Verify that the progress bar correctly reflects the number of chunks processed.
3.  **Concatenation**: Ensure the final audio file is a single complete WAV containing the whole text.
4.  **PDF utility**: Verify that dropping a PDF still extracts text into the editor without starting a "stream".

---

## Verification Plan

### Automated Tests
- No automated tests currently exist for this local studio. I will manually verify.

### Manual Verification
1.  **Terminal Check**: Start the server, generate a long text, and verify that the terminal shows lines like `[session-id] Progress: 1/5`.
2.  **UI Progress Bar**: Generate a voice in "Preset" mode and verify that a progress bar appears and moves from 0% to 100%.
3.  **Output Integrity**: Verify that the final audio generated in "Preset" mode is complete (all chunks merged) and playable.
4.  **Reader Mode**: Verify that the original "Doc Reader" mode still works correctly and shows its own progress bar.
