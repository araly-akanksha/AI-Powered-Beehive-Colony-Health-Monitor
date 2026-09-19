// Produces a plausible-looking log-mel spectrogram matrix for demo mode —
// rows = mel bands (low frequency at row 0), columns = time frames.
// When your backend is live, SpectrogramDisplay will render whatever it
// sends back instead (see services/api.js for the expected shape).

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
 * @param {boolean} opts.hardwareShift
 * @param {number} [bands=48]
 * @param {number} [frames=90]
 */
export function generateSyntheticSpectrogram({ state, seed = 1, hardwareShift = false }, bands = 48, frames = 90) {
  const rand = seededRandom(seed * 53 + (state === "present" ? 7 : 19));
  const matrix = [];

  // Queen piping sits roughly in mel-bands 14-24 out of 48 (~500-1500Hz)
  const pipingBandStart = hardwareShift ? 16 : 14;
  const pipingBandEnd = hardwareShift ? 26 : 24;

  for (let b = 0; b < bands; b++) {
    const row = [];
    for (let t = 0; t < frames; t++) {
      // 1/f-ish noise floor, louder at low bands (hive hum).
      let value = Math.max(0, 0.55 - b / bands) * (0.3 + rand() * 0.4);

      if (state === "present" && b >= pipingBandStart && b <= pipingBandEnd) {
        const wobble = Math.sin(t / 6 + b) * 0.15;
        const pulse = Math.sin(t / 3) > 0.2 ? 1 : 0.4;
        value += (0.55 + wobble) * pulse;
      }

      if (state === "absent") {
        // Sparse, disorganized flickers instead of a steady band.
        value += rand() > 0.93 ? rand() * 0.5 : 0;
      }

      row.push(Math.max(0, Math.min(1, value)));
    }
    matrix.push(row);
  }
  return matrix;
}
