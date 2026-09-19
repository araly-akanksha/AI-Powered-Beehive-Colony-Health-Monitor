import { API_BASE_URL } from "../config";

/**
 * ── Backend contract this frontend expects ───────────────────────────────
 *
 * POST {API_BASE_URL}/analyze
 *   Request:  multipart/form-data, field name "audio" = the .wav file
 *   Response: application/json
 *   {
 *     "hiveType": "known" | "unseen",
 *     "isOutOfDistribution": boolean,
 *     "models": [
 *       { "key": "baseline_cnn", "name": "Baseline CNN", "confidence": 0.54,
 *         "label": "Queen Present", "tag": "Inconclusive / Overfit" },
 *       { "key": "crnn", "name": "CRNN (GRU)", "confidence": 0.81,
 *         "label": "Queen Present", "tag": "Queen Present" },
 *       { "key": "dann", "name": "DANN (SOTA)", "confidence": 0.96,
 *         "label": "Queen Present", "tag": "Queen Present" }
 *     ],
 *     "metrics": {
 *       "spectralCentroidHz": 1842, "rms": 0.0051, "zcr": 0.061,
 *       "freqBand": "500 Hz – 2 kHz", "colonyStatus": "Normal"
 *     },
 *     "spectrogram": {
 *       // EITHER a base64-encoded PNG:
 *       "imageBase64": "<base64 PNG string>",
 *       // OR a raw log-mel matrix, rows = mel bands, cols = time frames:
 *       "bands": [[0.1, 0.2, ...], ...]
 *     }
 *   }
 *
 * GET {API_BASE_URL}/samples
 *   Response: [{ "id": "known-present", "title": "...", "hiveType": "known",
 *                "audioUrl": "/samples/hive1_known_present.wav" }, ...]
 *
 * If your FastAPI backend (backend/app.py) uses different field or route
 * names, either rename them to match above, or edit the two functions
 * below to map your actual response shape onto this structure — that's
 * the only place the rest of the app needs to know about it.
 * ──────────────────────────────────────────────────────────────────────
 */

async function parseJsonOrThrow(response) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || body.message || detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(`Backend error (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function analyzeAudio(file) {
  const formData = new FormData();
  formData.append("audio", file);

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData
  });

  return parseJsonOrThrow(response);
}

export async function fetchSampleClips() {
  const response = await fetch(`${API_BASE_URL}/samples`);
  return parseJsonOrThrow(response);
}
