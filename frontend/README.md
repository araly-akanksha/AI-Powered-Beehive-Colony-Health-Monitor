# Beehive Health Monitor — Frontend

React + Vite frontend for the beehive audio classification Demo Day app. This
is frontend only — it's built to sit in front of your existing FastAPI
backend, but also runs completely standalone in **demo mode** so you can
rehearse the full flow (upload → spectrogram → multi-model comparison → OOD
badge) before the backend is wired up, or as a safe fallback if it's
unreachable during the live presentation.

## Quick start

```bash
npm install
npm run dev
```

Open the printed local URL. It starts in demo mode by default — try the four
"Quick Demo Clips" buttons, or drag in any .wav/.mp3 file.

## Connecting your real backend

1. Copy `.env.example` to `.env` and set:
   ```
   VITE_DEMO_MODE=false
   ```
2. Make sure your FastAPI backend is running (default assumed: `localhost:8000`
   — change the proxy target in `vite.config.js` if yours differs).
3. Your backend needs to expose the two endpoints documented at the top of
   `src/services/api.js`:
   - `POST /api/analyze` — multipart upload, field name `audio`, returns the
     per-model predictions, bioacoustic metrics, and spectrogram data.
   - `GET /api/samples` — (optional) list of your real curated sample clips,
     if you want the "Quick Demo Clips" buttons to play real recordings from
     `backend/samples/` instead of the synthesized placeholder audio.

   If your backend already returns a different JSON shape, you don't need to
   change your backend — just edit `analyzeAudio()` / `fetchSampleClips()` in
   `src/services/api.js` to map your actual response onto the shape the rest
   of the app expects. That's the only file that needs to know about it.

## What's in demo mode vs. what needs your backend

| Feature | Demo mode | Live mode |
|---|---|---|
| Drag-and-drop upload, playback, validation | ✅ fully real | ✅ fully real |
| 4 curated Demo Day clips | Synthesized placeholder "buzz" audio + the exact confidence numbers from the project plan (54% / 81% / 96% on the unseen-queen clip, etc.) | Whatever your `/api/samples` + `/api/analyze` return |
| Multi-model comparison, headline badge, OOD badge, metrics panel, spectrogram | ✅ fully functional UI, fed by generated numbers | ✅ same UI, fed by your model's real output |
| Spectrogram | Procedurally drawn to look like a log-mel spectrogram (clearly labeled "Simulated for demo mode") | Renders your backend's real image or raw mel-band matrix |

Swap in your real 4 sample clips by either wiring up `GET /api/samples` (see
`src/services/api.js`), or, more simply, dropping real `.wav` files in
`public/samples/` and pointing `SAMPLE_CLIPS` in `src/services/mockData.js` at
them.

## Project structure

```
src/
  App.jsx                     — dashboard layout (the 4-panel grid + headline)
  config.js                   — demo mode toggle, API base URL
  components/
    Header.jsx                 — status pill + model-view selector
    Panel.jsx                  — shared card chrome for the 4 dashboard panels
    AudioUploader.jsx           — drag/drop zone + playback
    SampleClips.jsx              — the 4 Demo Day preset buttons
    BioacousticMetrics.jsx       — spectral centroid / RMS / ZCR / colony status
    SpectrogramDisplay.jsx       — renders backend image OR raw mel matrix
    ResultCard.jsx                — headline Queen Present/Absent badge
    ModelComparison.jsx            — Baseline CNN vs CRNN vs DANN, side by side
    OODBadge.jsx                    — in-distribution / out-of-distribution pill
  hooks/
    useAudioAnalysis.js         — all state: selected clip, staged progress, results
  services/
    api.js                     — real backend calls + the exact contract they expect
    mockData.js                — demo-mode data, incl. the curated 3-experiment numbers
  utils/
    syntheticAudio.js          — generates placeholder "hive buzz" wav clips
    syntheticSpectrogram.js    — generates a placeholder log-mel matrix
```

## Notes for Demo Day

- The "analyzing" banner steps through the same stage labels either way
  (feature extraction → spectrogram → each model), so the pacing feels the
  same whether you're on demo or live mode.
- The Model view selector in the header switches the headline badge and
  highlight between "Compare All 3" and any single model — handy if a
  professor asks "what does the baseline alone say here?"
- If your live backend goes down mid-presentation, set `VITE_DEMO_MODE=true`
  and reload — the curated clips reproduce your project plan's numbers
  exactly, so the story stays intact.
