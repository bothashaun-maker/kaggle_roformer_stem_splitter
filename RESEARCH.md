# Model and settings review — 7 November 2026

Reviewed all four repositories on the user's GitHub account. Movies contains the
H3, Movie Upscaler and Image Studio patterns; Music contains the working canvas,
chunked audio upload and asset workflow. The trading repository has no relevant
audio/runtime code. This repository contained the older multi-cell splitter.
Existing Music/Movies implementations were read, not changed.

## Corrections to the older notebook

- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` estimates vocals; running it again
  does not produce drums/bass/other. Its earlier “4-stem” label was incorrect.
- The six-stem command included `--two-stems=vocals`; that asks Demucs to write
  two outputs, even though the selected model supports six.
- `input_audio_path` is not the current Separator constructor's source argument.
  Passing the filename to `separate()` is required by that API.
- Generic filenames for vocals/karaoke/crowd/drums were not a verified catalog.
  Picking the first WAV found could select a complementary stem or old output.
- Repeated unpinned installs/downloads, independent output normalization, hidden
  stderr and shell fallbacks made failures and comparisons difficult to assess.

The original notebook is retained under `legacy/`. Its former root URL now opens
the new standalone ComfyUI notebook.

## Sources and selection

Primary sources consulted:

- [ZFTurbo pretrained-model tables](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/f0b527736e7da24eae5b561a2bb1efd0569ec757/docs/pretrained_models.md)
  distinguish vocal-only, instrument-specific and multi-instrument models and
  report named benchmark datasets.
- [Python Audio Separator](https://github.com/nomadkaraoke/python-audio-separator)
  supplies current model handling, execution policy and parameter definitions.
- [Kijai's ComfyUI MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer/tree/92c86854e6654f4aacc97484471af95c98ea16d4)
  was examined for the existing Comfy model layout and used as a local reference.
- [BS-RoFormer-Infer](https://github.com/openmirlab/bs-roformer-infer) recommends
  the fixed BS-RoFormer SW checkpoint for six sources and records its checksum.
- [Lucidrains BS-RoFormer](https://github.com/lucidrains/BS-RoFormer), installed
  as the pinned `bs-roformer==0.4.0`, provides both model architectures used here.
- [Demucs](https://github.com/adefossez/demucs) documents its four/six-source
  models, segment processing and the quality/runtime cost of repeated shifts.

| Job | Selected checkpoint | Reason and limitation |
| --- | --- | --- |
| Vocal/instrumental default | Kim MelBandRoformer | Documented vocal specialist; MIT model card; approximately 913 MB. Bleed/growls still need listening. |
| Alternative vocal extraction | Viperx BS-RoFormer `ep_317_sdr_12.9755` | Documented alternative with different errors; approximately 639 MB. The number in its filename is not a universal quality score. |
| Six instruments | BS-RoFormer SW Fixed | Explicit six-source checkpoint: bass, drums, other, vocals, guitar, piano; approximately 699 MB. No independent same-dataset comparison against all alternatives was performed here. |
| Four instruments | SW six-source result, with guitar/piano added to other | Coherent four-output grouping from one run. This is not a separate four-stem model. |

The ZFTurbo table reports Multisong vocal SDR around 10.98 for Kim and 10.87 for
Viperx; different datasets/model-name scores must not be compared as if they are
one leaderboard. SCNet XL IHF is also a strong documented four-source candidate
(its table reports 10.08 on MUSDB and 9.92 on Multisong). It is not included in
this first runtime: its architecture is different from the validated RoFormer
adapter. This release does not establish a universal “best model.”

Drum-kit-piece, organ, crowd, de-reverb and gender/lead/backing vocal specialists
are separate tasks, not capabilities silently attributed to these checkpoints.
No unverified `drumsep` package API, automatic cleanup cascade or expensive
ensemble is included. An exported stem can be selected as a later input for a
deliberate second vocal pass; repeated processing can also introduce artifacts.

## Settings chosen for Kaggle T4

- Batch size one, FP16 autocast on CUDA, PyTorch SDPA with math/memory-efficient
  eligibility on T4; no external FlashAttention package or Torch compilation.
- Approximately eight seconds per window. The sample count is rounded to the
  model's STFT hop: SW uses 352,768 samples; Kim/Viperx use 352,800 at 44.1 kHz.
  SW's upstream context is longer; this is an explicit memory/context tradeoff.
- Balanced overlap 2 (50%), Quality overlap 4 (75%), Maximum overlap 8 (87.5%).
  Kim/SW upstream configs use overlap 2; Viperx's published config uses overlap 4.
  Higher overlap adds compute and is not guaranteed to improve a particular song.
- Reflection/zero context padding, window fades and normalized overlap-add;
  CPU full-track buffers; one CUDA window at a time. A second T4's memory is not
  combined with GPU 0. The model process exits after each job to release memory.
- No individual normalization, shifting, cropping or lossy export. Model analysis
  is 44.1 kHz; final float WAVs retain original sample rate, channels and frame
  count. That preserves the file clock, not information the separator discarded.
- Two-source instrumental is the original-rate mix minus the restored vocal.
  Multi-source model outputs need not sum perfectly; the JSON reports residual RMS.
- Full-song jobs have no three-minute cutoff. RAM, disk and Kaggle session limits
  remain practical bounds. 7–8 minute T4 runs have not been timed here.

## Checkpoint/config compatibility

All three full checkpoints were downloaded locally and their bytes matched the
published SHA256 values in `stem_studio/catalog.json`. Each loaded strictly and
completed finite one-second CPU inference. The actual six-source job also passed
the production eight-second overlap-add/export/ZIP path on synthetic input.

Kim numbers its time/frequency layer pair `[0,1]`; the pinned library retains an
unused linear slot and numbers those same layers `[1,2]`. Only these root state
keys are renamed. Strict loading remains enabled. Against the independent Kijai
implementation using the same real checkpoint and input, the adapter's local
CPU output had maximum absolute difference **0.0**. No missing/unexpected weights
are ignored. Disabled training-only config fields are checked explicitly.

Kim's config was verified against the pinned ZFTurbo commit above. SW's config
comes from its pinned author/mirror revision; Viperx's config comes from the same
pinned model mirror as its weights. The exact source-config hashes and normalized
embedded configs are recorded in the catalog. Checkpoint URLs use immutable HF
revisions, sizes and SHA256 values. Library licenses do not relicense checkpoints;
original model terms remain applicable. Packs are private, not public mirrors.

## Storage and authentication

The three weights total **2,251,850,265 bytes (2.097 GiB)**. The builder copies
actual files into a durable output and verifies all of them before completion.
It then uses the pinned KaggleHub 1.0.3 SDK to upload only model-pack files and
issue an explicit private, create-only Dataset request. Failed submission retains
verified files. A receipt prevents a repeated successful submission in the same
session; existing Datasets are never versioned or overwritten by this builder.

The default immutable handle is
`janpierrrewiet/stem-studio-assets-v1-f6815b582a`. Rendering can auto-attach this
Dataset on Kaggle or read an attached saved pack directly. It verifies the pack
and reads weights from that location. Missing weights stop startup instead of
falling back to model-host downloads. Software and Cloudflare still install on
Kaggle each fresh session; this is not a fully offline software pack.

The builder uses available Kaggle authentication, with private notebook Secret
`KAGGLE_API_TOKEN` as a supported fallback. No credentials are embedded, printed
or committed. Dataset submission/processing and auto-attachment require validation
inside the user's authenticated Kaggle account; no submission occurred here.

## Validation limits

Local tests cover pack integrity/copying/reuse, SDK request shape and private flag,
failed submission preservation, interrupted upload recovery, offset/path/size
checks, overlap-add endpoints and final partial windows, mono/stereo clock and
float-WAV exports, four-source grouping, interruption and notebook/graph builds.
Real ComfyUI and headless Chromium checks are recorded in README.md.

Synthetic signal and CPU inference checks validate implementation and routing;
they do not measure music-separation quality, T4 VRAM/runtime, real Cloudflare
performance, or authenticated Kaggle Dataset creation. Listen to stems against
the source before deciding which vocal model/profile sounds best for a song.
