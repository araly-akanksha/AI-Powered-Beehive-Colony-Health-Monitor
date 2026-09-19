import { useCallback, useRef, useState } from "react";
import { DEMO_MODE } from "../config";
import { analyzeAudio } from "../services/api";
import { analyzeCuratedClip, analyzeUploadedFileDemo, synthesizeClipAudio, SAMPLE_CLIPS } from "../services/mockData";

const STAGE_SEQUENCE = [
  "Reading audio clip…",
  "Extracting bioacoustic features…",
  "Generating log-mel spectrogram…",
  "Running Baseline CNN…",
  "Running CRNN (GRU)…",
  "Running DANN…"
];

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useAudioAnalysis() {
  const [clip, setClip] = useState(null); // { name, audioUrl, source: 'upload'|'sample', sampleId? }
  const [stage, setStage] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | analyzing | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const objectUrlRef = useRef(null);

  const reset = useCallback(() => {
    setStatus("idle");
    setStage(null);
    setResult(null);
    setError(null);
  }, []);

  const runStagedProgress = useCallback(async () => {
    for (const label of STAGE_SEQUENCE) {
      setStage(label);
      // eslint-disable-next-line no-await-in-loop
      await wait(DEMO_MODE ? 260 : 180);
    }
  }, []);

  const analyzeFile = useCallback(
    async (file) => {
      reset();
      setStatus("analyzing");
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const audioUrl = URL.createObjectURL(file);
      objectUrlRef.current = audioUrl;
      setClip({ name: file.name, audioUrl, source: "upload" });

      try {
        const [analysis] = await Promise.all([
          DEMO_MODE ? analyzeUploadedFileDemo(file) : analyzeAudio(file),
          runStagedProgress()
        ]);
        setResult(analysis);
        setStatus("done");
      } catch (err) {
        setError(err.message || "Analysis failed.");
        setStatus("error");
      }
    },
    [reset, runStagedProgress]
  );

  const analyzeSample = useCallback(
    async (sampleId) => {
      const sample = SAMPLE_CLIPS.find((s) => s.id === sampleId);
      if (!sample) return;

      reset();
      setStatus("analyzing");

      try {
        let audioUrl;
        if (DEMO_MODE) {
          audioUrl = await synthesizeClipAudio(sample);
        } else if (sample.audioUrl) {
          audioUrl = sample.audioUrl;
        }
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        if (DEMO_MODE) objectUrlRef.current = audioUrl;
        setClip({ name: sample.title, audioUrl, source: "sample", sampleId, hiveType: sample.hiveType, note: sample.note });

        const [analysis] = await Promise.all([
          DEMO_MODE
            ? analyzeCuratedClip(sampleId)
            : fetch(audioUrl)
                .then((r) => r.blob())
                .then((blob) => analyzeAudio(new File([blob], `${sampleId}.wav`))),
          runStagedProgress()
        ]);
        setResult(analysis);
        setStatus("done");
      } catch (err) {
        setError(err.message || "Analysis failed.");
        setStatus("error");
      }
    },
    [reset, runStagedProgress]
  );

  return { clip, stage, status, result, error, analyzeFile, analyzeSample, reset };
}
