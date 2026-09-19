// Central place to point this frontend at your FastAPI backend.
//
// DEMO_MODE=true (default): no backend required. Every prediction is
// produced locally so you can rehearse the full Demo Day flow — upload,
// spectrogram, multi-model comparison, OOD badge — before your backend
// is wired up, or as a safe fallback if the backend is unreachable
// during the live presentation.
//
// DEMO_MODE=false: the app calls your FastAPI backend. See
// src/services/api.js for the exact request/response contract it
// expects, and README.md for how to line that up with your endpoints.
export const DEMO_MODE = String(import.meta.env.VITE_DEMO_MODE ?? "true") === "true";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

// Threshold below which the app treats a colony as "Absent / Distress"
// rather than "Present / Healthy" — only used to color-code demo output;
// your backend's own label field always wins when present.
export const PRESENT_LABEL = "Queen Present";
export const ABSENT_LABEL = "Queen Absent";
