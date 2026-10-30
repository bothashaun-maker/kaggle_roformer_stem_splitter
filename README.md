🎛️ Kaggle Roformer Stem SplitterHigh-quality AI stem separation running **100% free** on a Kaggle notebook — using the current state-of-the-art models (Mel-Band / BS Roformer + Demucs + DrumSep). Designed for dragging lossless stems straight into **REAPER** (or any DAW).
> Better than stock UVR5: the notebook runs the **Roformer transformer family**, which currently tops every source-separation benchmark.

---

## ✨ What you get

| Stem | Model used | Quality |
|---|---|---|
| Vocals / Instrumental | Mel-Band Roformer | ⭐ SOTA |
| Vocals (max quality) | BS-Roformer (`ep_317_sdr_12.9755`) | ⭐⭐ SOTA+ |
| 4-stem (vocals/drums/bass/other) | BS-Roformer | ⭐ SOTA |
| Guitar / Piano | Demucs `htdemucs_6s` | Very usable |
| Drums (dedicated stem) | Mel-Band Roformer drum model | ⭐ SOTA |
| Kick / Snare / Toms / Hi-hats / Cymbals | DrumSep (ResUNet) | Good* |
| Cleanup passes | Karaoke / Crowd Roformer | Situational |

*\*DrumSep is the oldest model in the chain — expect occasional toms/hats confusion on blast beats. Gate + transient shaping in your DAW cleans up the rest.*

---

## 🚀 Quick Start (no coding needed)

1. **Download** `kaggle_roformer_stem_splitter.ipynb` from this repo
2. Go to [kaggle.com](https://www.kaggle.com) → **File → Import Notebook** → upload the `.ipynb`
3. In the notebook: **Settings → Accelerator → GPU T4** (free tier, no credit card)
4. Upload your song (WAV / FLAC / MP3) via the **folder icon** in the right sidebar
5. **Run the cells top to bottom** (Shift+Enter, or Run → Run All)
6. Download **`all_stems.zip`** from the notebook's **Output** panel
7. Unzip → drag stems into your DAW project → they align at 0:00 automatically

> ⚠️ **Lossless tip:** feed it WAV/FLAC and you get lossless stems out. No compression anywhere in the chain.

---

## 🎚️ Which cells to run

| Cell | What it does | Needed? |
|---|---|---|
| 1–2 | Install + find your uploaded file | ✅ Always |
| 3 | Vocal / instrumental split (Roformer) | ✅ Recommended |
| 4 | Extra BS-Roformer vocals pass | Optional |
| 5 | 6-stem split incl. **guitar + piano** | Optional |
| 6 | List every available model | Handy |
| 7 | Targeted cleanup pass (e.g. isolate organ from "other") | Optional |
| 8a/8b | Drums stem + kit-piece split (kick/snare/toms/hats/cymbals) | Optional |
| 9 | Zip everything for download | ✅ Always |

All optional cells can be skipped — the notebook never dead-ends (drum cell falls back to Demucs automatically).

---

## 🥁 Recommended chain (full workflow)

```
Full mix (WAV/FLAC)
 ├─ Cell 3  → vocals.wav / instrumental.wav        (Mel-Band Roformer)
 ├─ Cell 5  → guitar.wav / piano.wav / other.wav   (Demucs 6-stem)
 ├─ Cell 7  → organ isolated from "other"          (cleanup pass)
 ├─ Cell 8a → drums.wav                            (Roformer drum model)
 └─ Cell 8b → kick / snare / toms / hats / cymbals (DrumSep on the drums stem)
```

---

## 🎸 Into REAPER

1. Unzip `all_stems.zip`
2. Drag all stems into one project — they start at 0:00 from the same source clock, so everything stays in sync
3. Mix freely:
   - **Kick** → gate + transient designer
   - **Snare** → saturate/compress solo
   - **Cymbals** → separate bus so double-kick stays aggressive while the wash stays clean
   - **Vocals** → stay dry and forward; no need to fight reverb bleed

### Quality tricks
- **Double-pass trick:** if a stem still bleeds (e.g. guitar residue in vocals), run *that stem* back through the vocal model again
- **Organ / synth rescue:** organs usually land in "other" — use cell 7 to feed "other" through a specialist model
- **Newer models:** check cell 6's model list occasionally; `audio-separator` adds new Roformer checkpoints regularly

---

## 📋 Requirements

- Free Kaggle account (GPU T4 tier — roughly 30h/week free)
- Your song as WAV/FLAC/MP3
- Nothing installed on your own PC — it all runs in the cloud

---

## 🙏 Credits

- [audio-separator](https://github.com/damiafuentes/PythonAudioSeparator) — model manager & separator
- [BS-Roformer / Mel-Band Roformer](https://github.com/lucidrains/BS-RoFormer) — Lucidrains + kuielab
- [MusicSourceSeparationTraining](https://github.com/ZFTurba/MusicSourceSeparationTraining) — ZFTurba's trained checkpoints
- [Demucs](https://github.com/adefossez/demucs) — Meta/Facebook Research
- [DrumSep](https://github.com/inagoy/drumsep) — inagoy

## ⚖️ License & legal

Notebook code: use freely. Separating stems of copyrighted music for your own remix/mixing work is generally fine; distributing separated copyrighted material is not. Respect the rights of original owners.
```

**GitHub in 3 moves:** New repository → **Add file → Create new file** → name it `README.md` → paste this → **Commit changes**. Lekker. ✅
