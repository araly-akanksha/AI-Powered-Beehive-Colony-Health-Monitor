// ============================================================
// src/api/beehive.js — API client for the FastAPI backend
// ============================================================

import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({ baseURL: BASE_URL, timeout: 60000 });

/** Check if the backend + model are available */
export async function checkHealth() {
  const { data } = await api.get('/health');
  return data;
}

/** Get list of pre-loaded demo clips */
export async function getSamples() {
  const { data } = await api.get('/samples');
  return data.samples;
}

/**
 * Upload a WAV file and get a prediction
 * @param {File} file - WAV file object
 * @returns {Promise<Object>} - { label, confidence, all_confidences, spectrogram_b64, ... }
 */
export async function predictFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/**
 * Predict on a pre-loaded sample clip by name
 * @param {string} clipName - filename in backend/samples/
 */
export async function predictSample(clipName) {
  const { data } = await api.post(`/predict-sample/${clipName}`);
  return data;
}
