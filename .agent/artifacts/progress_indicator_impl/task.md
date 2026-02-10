# Task Checklist: Implement Progress Indicator

- [x] Research Implementation [x]
    - [x] List project directory
    - [x] Analyze `app.py` generation logic
    - [x] Analyze `script.js` to see how it handles the generation request
- [x] Create Implementation Plan [x]
- [x] Implement Backend Progress Tracking [x]
    - [x] Add terminal prints in `generation_worker`
    - [x] Add `/api/stream/concatenate` endpoint
- [x] Implement Frontend Progress UI [x]
    - [x] Add progress bar to `index.html` (and remove `stream-card`)
    - [x] Style the progress bar in `style.css`
- [x] Update Frontend Logic [x]
    - [x] Unify `generateBtn` to use sessions and polling
    - [x] Remove Reader streaming logic (keep PDF extraction)
- [x] Verification [x]
    - [ ] Test the generation flow and verify the progress indicator works correctly
