import { generateHiveClip } from "../utils/syntheticAudio";
import { generateSyntheticSpectrogram } from "../utils/syntheticSpectrogram";
import { PRESENT_LABEL, ABSENT_LABEL } from "../config";

// The three models from the dissertation's experiment sequence, in the
// order they should always be displayed.
export const MODELS = [
  { key: "baseline_cnn", name: "Baseline CNN", tagline: "No domain adaptation" },
  { key: "crnn", name: "CRNN (GRU)", tagline: "Adds temporal context" },
  { key: "dann", name: "DANN (SOTA)", tagline: "Domain-adversarial, hardware-invariant" }
];

// The four curated Demo Day clips (Upgrade 4). Swap `audioUrl` for a real
// path under backend/samples/ once you're on the live backend — see
// SampleClips.jsx and README.md.
export const SAMPLE_CLIPS = [
  {
    id: "known-present",
    label: "Known Queen",
    title: "Known Hive — Queen Present",
    hiveType: "known",
    groundTruth: "present",
    seed: 11,
    hardwareShift: false,
    audioUrl: "/samples/known_present.wav",
    note: "In-distribution. All three models should agree easily."
  },
  {
    id: "known-absent",
    label: "Known Absent",
    title: "Known Hive — Queen Absent",
    hiveType: "known",
    groundTruth: "absent",
    seed: 23,
    hardwareShift: false,
    audioUrl: "/samples/known_absent.wav",
    note: "In-distribution negative case — quieter, no organized piping."
  },
  {
    id: "unseen-present",
    label: "Unseen Queen",
    title: "Unseen Colony — Queen Present",
    hiveType: "unseen",
    groundTruth: "present",
    seed: 37,
    hardwareShift: false,
    audioUrl: "/samples/unseen_present.wav",
    note: "Cross-hive generalization test. Baseline CNN typically struggles here."
  },
  {
    id: "unseen-absent",
    label: "Unseen Absent",
    title: "Unseen Colony — IoT MEMS Mic",
    hiveType: "unseen",
    groundTruth: "absent",
    seed: 41,
    hardwareShift: true,
    audioUrl: "/samples/unseen_absent.wav",
    note: "Different microphone hardware — tests hardware-invariant transfer."
  }
];

// Fixed, hand-picked model outputs per clip so the numbers you see in
// rehearsal match the ones in your project plan. Uploaded files that
// aren't one of the four presets get a plausible generated result instead
// (see analyzeAudioDemo below).
const CURATED_RESULTS = {
  "known-present": {
    baseline_cnn: { confidence: 0.92, label: PRESENT_LABEL, tag: "Confident" },
    crnn: { confidence: 0.95, label: PRESENT_LABEL, tag: "Confident" },
    dann: { confidence: 0.97, label: PRESENT_LABEL, tag: "Confident" },
    metrics: { spectralCentroidHz: 1842, rms: 0.0051, zcr: 0.061, freqBand: "500 Hz – 2 kHz", colonyStatus: "Normal" }
  },
  "known-absent": {
    baseline_cnn: { confidence: 0.88, label: ABSENT_LABEL, tag: "Confident" },
    crnn: { confidence: 0.91, label: ABSENT_LABEL, tag: "Confident" },
    dann: { confidence: 0.95, label: ABSENT_LABEL, tag: "Confident" },
    metrics: { spectralCentroidHz: 1024, rms: 0.0029, zcr: 0.104, freqBand: "300 Hz – 1.2 kHz", colonyStatus: "Distress" }
  },
  "unseen-present": {
    baseline_cnn: { confidence: 0.54, label: PRESENT_LABEL, tag: "Inconclusive / Overfit" },
    crnn: { confidence: 0.81, label: PRESENT_LABEL, tag: "Queen Present" },
    dann: { confidence: 0.96, label: PRESENT_LABEL, tag: "Queen Present" },
    metrics: { spectralCentroidHz: 1793, rms: 0.0047, zcr: 0.067, freqBand: "500 Hz – 2 kHz", colonyStatus: "Normal" }
  },
  "unseen-absent": {
    baseline_cnn: { confidence: 0.49, label: ABSENT_LABEL, tag: "Inconclusive / Overfit" },
    crnn: { confidence: 0.7, label: ABSENT_LABEL, tag: "Uncertain" },
    dann: { confidence: 0.91, label: ABSENT_LABEL, tag: "Queen Absent" },
    metrics: { spectralCentroidHz: 2211, rms: 0.0033, zcr: 0.098, freqBand: "600 Hz – 2.2 kHz", colonyStatus: "Distress" }
  }
};

function hashStringToSeed(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h) || 1;
}

/**
 * Build a demo-mode "analysis result" for one of the four curated clips.
 */
export async function analyzeCuratedClip(clipId) {
  const clip = SAMPLE_CLIPS.find((c) => c.id === clipId);
  if (!clip) throw new Error(`Unknown demo clip: ${clipId}`);
  const curated = CURATED_RESULTS[clipId];

  const spectrogram = generateSyntheticSpectrogram(
    { state: clip.groundTruth, seed: clip.seed, hardwareShift: clip.hardwareShift }
  );

  return {
    hiveType: clip.hiveType,
    isOutOfDistribution: clip.hiveType === "unseen",
    models: MODELS.map((m) => ({ ...m, ...curated[m.key] })),
    metrics: curated.metrics,
    spectrogram: { bands: spectrogram }
  };
}

/**
 * Build a plausible (non-curated) demo result for an arbitrary uploaded
 * file, so the "drop your own clip" path still demonstrates something
 * coherent instead of erroring out when there's no backend yet.
 */
export async function analyzeUploadedFileDemo(file) {
  const seed = hashStringToSeed(file.name + file.size);
  const rand = ((seed % 1000) / 1000);
  const groundTruth = rand > 0.45 ? "present" : "absent";
  const hiveType = rand > 0.7 ? "unseen" : "known";

  const spectrogram = generateSyntheticSpectrogram({ state: groundTruth, seed, hardwareShift: false });

  const baseConfidence = hiveType === "unseen" ? 0.5 + rand * 0.15 : 0.85 + rand * 0.1;
  const label = groundTruth === "present" ? PRESENT_LABEL : ABSENT_LABEL;

  const models = MODELS.map((m, idx) => {
    const bump = idx * (hiveType === "unseen" ? 0.18 : 0.04);
    const confidence = Math.min(0.98, baseConfidence + bump);
    return {
      ...m,
      confidence,
      label,
      tag: confidence < 0.6 ? "Inconclusive / Overfit" : confidence < 0.8 ? "Uncertain" : "Confident"
    };
  });

  return {
    hiveType,
    isOutOfDistribution: hiveType === "unseen",
    models,
    metrics: {
      spectralCentroidHz: Math.round(1000 + rand * 1200),
      rms: Number((0.002 + rand * 0.005).toFixed(4)),
      zcr: Number((0.05 + rand * 0.06).toFixed(3)),
      freqBand: "500 Hz – 2 kHz",
      colonyStatus: groundTruth === "present" ? "Normal" : "Distress"
    },
    spectrogram: { bands: spectrogram }
  };
}

/**
 * Synthesize the playable audio for a curated demo clip on demand.
 */
export function synthesizeClipAudio(clip) {
  return generateHiveClip({ state: clip.groundTruth, seed: clip.seed, hardwareShift: clip.hardwareShift });
}
