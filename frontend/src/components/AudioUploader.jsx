import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = [".wav", ".mp3", ".flac", ".ogg"];

export default function AudioUploader({ clip, onFileSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const inputRef = useRef(null);

  const validateAndSelect = useCallback(
    (file) => {
      setValidationError(null);
      if (!file) return;

      const nameLower = file.name.toLowerCase();
      const hasValidExtension = ACCEPTED_TYPES.some((ext) => nameLower.endsWith(ext));
      if (!hasValidExtension) {
        setValidationError(`Unsupported file type. Use one of: ${ACCEPTED_TYPES.join(", ")}`);
        return;
      }
      if (file.size === 0) {
        setValidationError("That file looks empty — pick another clip.");
        return;
      }
      if (file.size > 20 * 1024 * 1024) {
        setValidationError("Clip is larger than 20MB — trim it before uploading.");
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      validateAndSelect(file);
    },
    [disabled, validateAndSelect]
  );

  return (
    <div className="audio-uploader">
      <div
        className={`drop-zone ${isDragging ? "drop-zone--active" : ""} ${disabled ? "drop-zone--disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) inputRef.current?.click();
        }}
        aria-label="Drag and drop a .wav clip, or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="drop-zone__input"
          onChange={(e) => validateAndSelect(e.target.files?.[0])}
          disabled={disabled}
        />
        <p className="drop-zone__label">Drag &amp; drop a .wav clip</p>
        <p className="drop-zone__hint">or click to browse · {ACCEPTED_TYPES.join(", ")}</p>
      </div>

      {validationError && <p className="drop-zone__error">{validationError}</p>}

      {clip?.audioUrl && (
        <div className="audio-playback">
          <audio controls src={clip.audioUrl} key={clip.audioUrl} />
          <span className="audio-playback__name" title={clip.name}>{clip.name}</span>
        </div>
      )}
    </div>
  );
}
