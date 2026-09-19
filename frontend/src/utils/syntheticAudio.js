// Generates short, plausible "hive buzz" audio entirely in the browser.
//
// This exists ONLY for demo mode, so the four preset clips are audibly
// playable without shipping real recordings or a backend. A present-queen
// clip layers a steady piping tone (with harmonics in the ~500-1500Hz band
// your dissertation focuses on) over a broadband hum; an absent-queen clip
// is quieter and more erratic, with no organized piping tone. Swap these
// out for your real TBON .wav files the moment your backend is connected —
// see README.md.

const SAMPLE_RATE = 22050;
const DURATION_S = 2.0;

function encodeWav(audioBuffer) {
  const numChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const samples = audioBuffer.getChannelData(0);
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function seededRandom(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/**
 * @param {object} opts
 * @param {"present"|"absent"} opts.state
 * @param {number} opts.seed
 * @param {boolean} opts.hardwareShift - simulate a MEMS-mic style frequency emphasis
 * @returns {Promise<string>} object URL for a WAV blob
 */
export async function generateHiveClip({ state, seed = 1, hardwareShift = false }) {
  const length = Math.floor(SAMPLE_RATE * DURATION_S);
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new OfflineCtx(1, length, SAMPLE_RATE);
  const rand = seededRandom(seed * 97 + (state === "present" ? 13 : 29));

  const master = ctx.createGain();
  master.gain.value = state === "present" ? 0.5 : 0.28; // present hives read ~44% louder
  master.connect(ctx.destination);

  // Broadband hive hum: filtered noise, present in both states.
  const noiseBuffer = ctx.createBuffer(1, length, SAMPLE_RATE);
  const noiseData = noiseBuffer.getChannelData(0);
  for (let i = 0; i < length; i++) noiseData[i] = rand() * 2 - 1;
  const noiseSrc = ctx.createBufferSource();
  noiseSrc.buffer = noiseBuffer;
  noiseSrc.loop = true;
  const noiseFilter = ctx.createBiquadFilter();
  noiseFilter.type = "bandpass";
  noiseFilter.frequency.value = hardwareShift ? 260 : 220;
  noiseFilter.Q.value = 0.7;
  const noiseGain = ctx.createGain();
  noiseGain.gain.value = state === "present" ? 0.35 : 0.55;
  noiseSrc.connect(noiseFilter).connect(noiseGain).connect(master);
  noiseSrc.start(0);

  if (state === "present") {
    // Queen piping: fundamental + harmonics inside ~500-1500Hz, steady envelope.
    const fundamental = hardwareShift ? 430 : 400;
    [1, 2, 3].forEach((harmonic, idx) => {
      const osc = ctx.createOscillator();
      osc.type = idx === 0 ? "sine" : "triangle";
      osc.frequency.value = fundamental * harmonic;
      const g = ctx.createGain();
      const level = 0.18 / (idx + 1);
      g.gain.setValueAtTime(0, 0);
      g.gain.linearRampToValueAtTime(level, 0.15);
      g.gain.setValueAtTime(level, DURATION_S - 0.2);
      g.gain.linearRampToValueAtTime(0, DURATION_S);
      // Gentle periodic tremor, like a live piping call.
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 4.5;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = level * 0.25;
      lfo.connect(lfoGain).connect(g.gain);
      lfo.start(0);
      osc.connect(g).connect(master);
      osc.start(0);
    });
  } else {
    // Absent: erratic amplitude modulation, no organized tone — restlessness.
    const modBuffer = ctx.createBuffer(1, length, SAMPLE_RATE);
    const modData = modBuffer.getChannelData(0);
    let envelope = 0.4;
    for (let i = 0; i < length; i++) {
      envelope += (rand() - 0.5) * 0.02;
      envelope = Math.max(0.15, Math.min(0.8, envelope));
      modData[i] = envelope;
    }
    const modSrc = ctx.createBufferSource();
    modSrc.buffer = modBuffer;
    const modGain = ctx.createGain();
    modGain.gain.value = 0.4;
    const carrier = ctx.createOscillator();
    carrier.type = "sawtooth";
    carrier.frequency.value = 180;
    const carrierGain = ctx.createGain();
    modSrc.connect(modGain);
    carrier.connect(carrierGain);
    modGain.connect(carrierGain.gain);
    carrierGain.connect(master);
    modSrc.start(0);
    carrier.start(0);
  }

  const rendered = await ctx.startRendering();
  const blob = encodeWav(rendered);
  return URL.createObjectURL(blob);
}
