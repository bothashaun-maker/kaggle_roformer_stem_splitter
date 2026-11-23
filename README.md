# Stem Studio — Kaggle + ComfyUI

One startup cell opens a normal ComfyUI canvas through Cloudflare. Upload your
original WAV, select the song, press Run, audition stems and download the ZIP.
Everything runs on Kaggle. No local software is needed.

## First time: save the models

1. Import [Stem_Splitter_Model_Packs.ipynb](Stem_Splitter_Model_Packs.ipynb) into Kaggle.
2. Set **Accelerator: None**, **Internet ON**, then run its single cell.
3. Wait for **PRIVATE DATASET SUBMITTED**, open the printed link and wait for Kaggle processing.

The private named Dataset is **Stem Studio Models f6815b582a**, with about **2.1 GiB**
of verified weights. Its handle is `janpierrrewiet/stem-studio-assets-v1-f6815b582a`.
Transfers go directly between the model hosts and Kaggle. Nothing is downloaded to
or uploaded from your PC. The builder never updates an existing Dataset. If an
authentication error occurs, enable the private Kaggle Secret `KAGGLE_API_TOKEN`
and rerun the same cell. Verified model files are reused. Never paste a token into code.

You can also **Save Version with output files** and attach the saved pack under
Add Input → Notebook Outputs. A saved draft alone does not preserve the files.

## Every song session

1. Import [Stem_Splitter_ComfyUI_Kaggle.ipynb](Stem_Splitter_ComfyUI_Kaggle.ipynb).
2. Choose **GPU T4 ×2**, **Internet ON**, and run its single startup cell.
3. The saved Dataset is auto-attached when supported by Kaggle. Alternatively use
   **Add Input → Your Datasets → Stem Studio Models f6815b582a** before startup.
4. Open the printed Six stems, Four stems or Vocals canvas link.
5. Use **Upload song** or the printed upload link, then select your file on the canvas.
6. Press **Run once**. After completion use **Download all stems ZIP** on the Export node.

Keep the startup cell running while using the canvas. Stop another music/movie/image
startup cell before opening this studio on port 8188. A second T4 does not pool its
memory with GPU 0. CPU work for decoding/resampling/export is normal on Kaggle.
Model inference requires CUDA and never silently falls back to CPU.

## Models and starting settings

| Canvas | Model | Outputs |
| --- | --- | --- |
| Vocals | Kim Mel-Band RoFormer | vocals, instrumental |
| Vocals alternative | Viperx BS-RoFormer 12.9755 | vocals, instrumental |
| Six stems | BS-RoFormer SW Fixed | vocals, drums, bass, guitar, piano, other |
| Four stems | Same SW model, guitar/piano grouped into other | vocals, drums, bass, other |

Start with **Balanced**: roughly 8-second windows, batch size one, overlap 2.
**Quality** uses overlap 4 (the published Viperx setting). Maximum uses overlap 8
and takes longer. Higher overlap does not guarantee better sound. Compare a few
sections by changing the audition start/stem without repeating model inference.
Completed jobs are reused when the source bytes/model/settings match.

Large WAVs upload in resumable **4-MiB chunks**. No file conversion is required.
An optional attached path accepts files under `/kaggle/input` or Studio's output
folder, including a previously exported stem for a deliberate cleanup pass.

The six-node graph keeps separate stages for source, settings, separation,
export and audition. All three canvases are also installed under ComfyUI's Workflows menu.

## Files in REAPER

The ZIP contains **32-bit float WAVs** and `result.json`. Import all WAVs at 0:00.
Each retains the source's exact frame count, sample rate and mono/stereo channels.
There is no individual stem normalization or lossy encoding. Internal model analysis
uses 44.1 kHz. Export does not restore detail discarded by separation. Bleed and
artifacts can remain, especially for guitar/piano and difficult vocals.

Each job has its own folder under `/kaggle/working/stem_studio/jobs/`, with stems,
ZIP, provenance, progress/ETA, device and GPU peak-memory diagnostics. Download the
ZIP or **Save Version with output files** before ending your session. The model
Dataset preserves weights, not your generated stems. Software still installs over
Kaggle's connection on fresh sessions. The saved Dataset avoids model downloads.

## Research, validation and rebuilding

[RESEARCH.md](RESEARCH.md) records sources, model comparisons, tradeoffs and fixes
to the old notebook's incorrect model/API assumptions. No model is declared best
for every song. Specialist drum-kit/organ/cleanup models are not misrepresented
as features of this three-checkpoint bundle.

Local checks: all three published checkpoints verified and loaded strictly.
CPU inference was finite for each. The Kim adapter matched the independent
reference implementation exactly on the test input. A real six-source CPU job
completed through export/ZIP. Unit tests cover pack integrity/private SDK creation, upload
recovery, sample alignment, float export and notebook/graph generation: **16 tests
passed**. The real pinned ComfyUI server registered all six graph stages.
Headless Chromium verified a **75,146,444-byte WAV upload**, live song selection,
canvas Run using genuine saved separation results, audition/export completion
and ZIP download, with no page errors. Original/uploaded WAV bytes and downloaded
ZIP bytes were compared directly. See [CHECKS.md](CHECKS.md) for scope.

These checks do **not** establish full-song quality, T4 VRAM/runtime, real
Cloudflare transfer performance, or authenticated Kaggle Dataset creation and
auto-attachment. Those require live Kaggle validation. No Kaggle GPU time was used here.

```bash
python tools/build_notebooks.py
python -m unittest discover -s tests -v
# With a running local ComfyUI test server and synthetic fixtures:
node tools/browser_check.js
```

The old root [notebook URL](kaggle_roformer_stem_splitter.ipynb) now aliases the new
ComfyUI notebook. The original multi-cell file is preserved under `legacy/`.
Model weights are not stored in GitHub. Source libraries and model terms retain
their upstream licenses. References and immutable model URLs are in the catalog.
