# Local release checks

Executed on 7 November 2026. No user recording or Kaggle GPU time was used.

- Three checkpoints: full published sizes/SHA256 verified (2,251,850,265 bytes).
- Full model pack: CPU builder copied actual weights, wrote completion manifest,
  and verified all three. No symlinks in saved outputs.
- Real pinned `bs-roformer==0.4.0`: strict state loading and finite CPU inference
  for SW, Kim and Viperx. `tools/check_models.py` reproduces this using a saved pack.
- Kim adapter/reference comparison: identical CPU samples, maximum difference 0.0.
- Actual SW six-source production job: eight-second windows, overlap-add, sample
  rate restoration, six float WAVs and ZIP passed on a 48-kHz synthetic recording.
  All six outputs retained its exact 14,413-frame count and stereo layout.
- 16 automated tests passed, covering storage, private SDK request fields/signature,
  failed-submission preservation, upload resume/rejection, partial-window coverage,
  float exports, source-clock preservation, four-output grouping, worker error
  delivery to the canvas, and builds. All three notebooks passed v4 schema validation.
- ComfyUI `423cbd20511becb1f16122ab518e92e58bfa03d9` with frontend 1.57.4 started on
  CPU and registered all custom stages plus native PreviewAudio.
- Chromium: ordinary six-node canvas loaded, upload button displayed, a
  75,146,444-byte synthetic WAV uploaded in 4-MiB requests, and the song selector
  updated the actual backend Run prompt. A second short synthetic recording
  matched the completed actual-model job and reused its saved stems. Run completed,
  audition and export output nodes executed, Download ZIP appeared, and the ZIP
  downloaded successfully. No uncaught page errors. The test checks the host has
  initialized its compatibility API before inspecting it.
- Startup now waits for the pinned frontend's initial workflow restoration and
  preserves its current workflow hash/router state when removing the one-time
  canvas flag. Refreshed song selection marks the graph changed so it reaches
  prompt serialization. The upload button is installed after graph configuration.

The browser-uploaded large WAV bytes and downloaded ZIP bytes were compared
directly with their originals; ZIP integrity and all six WAV entries passed.

Package installation retained Torch 2.11 CPU for these local tests. A CPU
Torchvision build was used locally. The Kaggle launcher instead constrains its
installed Torch/Torchvision/Torchaudio/NumPy/SciPy versions; it does not replace
Kaggle's CUDA stack. Production rejects a missing CUDA device.

Not established by these checks: actual T4 VRAM/time, musical separation quality,
7–8 minute model inference on Kaggle, real Cloudflare upload reliability, or
authenticated Dataset creation/processing/auto-attachment. The SDK Dataset request
was mocked, not sent. Synthetic endpoint tests do not prove musical accuracy.
