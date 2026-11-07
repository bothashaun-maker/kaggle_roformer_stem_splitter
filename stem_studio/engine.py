"""File-based jobs and bounded RoFormer inference. Original media remains read-only."""
from pathlib import Path
import contextlib
import gc
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
import soundfile as sf
from .assets import CATALOG, FINGERPRINT, sha256, valid

PIPELINE = 'stem-studio-v1-cpu-ola-no-normalization'
PROFILES = {'Balanced': 2, 'Quality': 4, 'Maximum overlap (slower)': 8}


def load_audio(path):
    """Decode without modifying the upload; retain its rate/frame count for export."""
    path = Path(path)
    try:
        data, rate = sf.read(path, dtype='float32', always_2d=True)
    except RuntimeError:
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels', '-of', 'json', str(path)], text=True)
        info = json.loads(probe)['streams'][0]
        rate, channels = int(info['sample_rate']), int(info['channels'])
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], stderr=subprocess.PIPE)
        data = np.frombuffer(raw, np.float32).reshape(-1, channels).copy()
    if rate < 8000 or rate > 192000 or data.shape[1] not in (1, 2):
        raise ValueError('Use mono/stereo audio at 8–192 kHz; multichannel audio is not silently downmixed')
    if not len(data) or not np.isfinite(data).all() or not np.any(data):
        raise ValueError('Audio is empty, silent or nonfinite')
    return np.ascontiguousarray(data.T), rate


def resample(audio, old_rate, new_rate, length=None):
    from scipy.signal import resample_poly
    if old_rate != new_rate:
        divisor = math.gcd(old_rate, new_rate)
        audio = resample_poly(audio, new_rate // divisor, old_rate // divisor, axis=-1)
    if length is not None:
        audio = audio[..., :length]
        if audio.shape[-1] < length:
            audio = np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, length - audio.shape[-1])])
    return np.ascontiguousarray(audio, dtype=np.float32)


def construct_model(spec, weights):
    import torch
    from bs_roformer import BSRoformer, MelBandRoformer
    cls = MelBandRoformer if spec['architecture'] == 'mel' else BSRoformer
    params = dict(spec['config']['model'])
    # Only non-architectural training extensions are omitted; strict state loading
    # below verifies every parameter. These upstream extensions are all disabled.
    allowed_ignored = {'linear_transformer_depth': 0, 'mlp_expansion_factor': 4,
                       'use_torch_checkpoint': False, 'skip_connection': False}
    accepted = inspect.signature(cls.__init__).parameters
    for key in list(params):
        if key not in accepted:
            if key not in allowed_ignored or params[key] != allowed_ignored[key]:
                raise ValueError('Unsupported model architecture parameter: ' + key)
            del params[key]
    for key in ('freqs_per_bands', 'multi_stft_resolutions_window_sizes'):
        if key in params:
            params[key] = tuple(params[key])
    # Kim's original config omits this but its checkpoint has no linear layers.
    if spec['architecture'] == 'mel':
        params.setdefault('linear_transformer_depth', 0)
        params['match_input_audio_length'] = True
    model = cls(**params).eval()
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in state:
        state = state['state_dict']
    if spec['architecture'] == 'mel':
        # Kim names each [time, frequency] pair [0, 1]. bs-roformer 0.4.0
        # retains a disabled linear-attention slot, naming the same pair [1, 2].
        # Rename only these root keys; strict loading still consumes all weights.
        state = {re.sub(r'^layers\.(\d+)\.([01])\.',
                        lambda m: f'layers.{m[1]}.{int(m[2])+1}.', key): value
                 for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def overlap_add(mix, model, stems, chunk_size, overlap, device, progress=None, interrupt=None):
    """Full-track buffers stay on CPU; only one model window occupies CUDA."""
    import torch
    import torch.nn.functional as F
    if overlap not in (2, 4, 8) or chunk_size < 4096:
        raise ValueError('Invalid overlap or chunk size')
    step = chunk_size // overlap
    border = chunk_size - step
    # Reflection gives both ends context; zero-pad very short inputs safely.
    padding = 'reflect' if mix.shape[-1] > border else 'constant'
    padded = np.pad(mix, ((0, 0), (border, border)), mode=padding)
    total = padded.shape[-1]
    result = np.zeros((stems, 2, total), np.float32)
    counter = np.zeros(total, np.float32)
    window = np.ones(chunk_size, np.float32)
    fade = chunk_size // 10
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    positions = list(range(0, total, step))
    began = time.monotonic()
    with torch.inference_mode():
        for number, start in enumerate(positions, 1):
            if interrupt:
                interrupt()
            count = min(chunk_size, total - start)
            part = torch.from_numpy(padded[:, start:start + count].copy())
            if count < chunk_size:
                part = F.pad(part, (0, chunk_size - count))
            amp = torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else contextlib.nullcontext()
            with amp:
                prediction = model(part.unsqueeze(0).to(device))[0]
            prediction = prediction.float().cpu().numpy()
            if prediction.ndim == 2:
                prediction = prediction[None]
            if prediction.shape[:2] != (stems, 2) or prediction.shape[-1] < count or not np.isfinite(prediction).all():
                raise ValueError('Invalid model output shape or nonfinite samples')
            weight = window[:count].copy()
            if start == 0:
                weight[:fade] = 1
            if start + chunk_size >= total:
                weight[-min(fade, count):] = 1
            result[..., start:start + count] += prediction[..., :count] * weight
            counter[start:start + count] += weight
            if progress:
                elapsed = time.monotonic() - began
                progress({'stage': 'separation', 'chunk': number, 'chunks': len(positions),
                          'elapsed_seconds': round(elapsed, 1),
                          'eta_seconds': round(elapsed / number * (len(positions) - number), 1)})
    if np.any(counter <= 0):
        raise ValueError('Overlap-add left uncovered samples')
    result /= counter
    return result[..., border:border + mix.shape[-1]]


def export_stems(original, rate, estimates, model_rate, names, mode, directory):
    """Export aligned float WAVs without individual stem normalization/clipping."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    restored = {name: resample(stem, model_rate, rate, original.shape[-1])
                for name, stem in zip(names, estimates)}
    if original.shape[0] == 1:
        restored = {name: value.mean(axis=0, keepdims=True) for name, value in restored.items()}
    if len(names) == 1:
        restored['instrumental'] = original - restored['vocals']
    elif mode == 'Four stems':
        restored['other'] = restored['other'] + restored.pop('guitar') + restored.pop('piano')
    files = {}
    for name, value in restored.items():
        if value.shape != original.shape or not np.isfinite(value).all():
            raise ValueError('Stem failed sample-count/channel validation: ' + name)
        path = directory / (name + '.wav')
        sf.write(path, value.T, rate, subtype='FLOAT')
        files[name] = {'path': str(path), 'sha256': sha256(path),
                       'frames': value.shape[-1], 'sample_rate': rate, 'channels': value.shape[0],
                       'peak': float(np.max(np.abs(value)))}
    combined = sum(restored.values())
    return files, {'sum_residual_rms': float(np.sqrt(np.mean((original - combined).astype(np.float64)**2))),
                   'format': '32-bit float WAV', 'per_stem_normalization': False}


def run_job(source, model_key, mode, profile, models, output_root, allow_cpu=False, interrupt=None):
    import fcntl
    if model_key not in CATALOG or profile not in PROFILES:
        raise ValueError('Unknown model or profile')
    if mode not in ('Vocals + instrumental', 'Four stems', 'Six stems'):
        raise ValueError('Unknown output mode')
    if (model_key == 'sw6') != (mode in ('Four stems', 'Six stems')):
        raise ValueError('Choose a vocal model for two stems, or BS RoFormer SW for four/six stems')
    spec = CATALOG[model_key]
    source = Path(source).resolve(strict=True)
    weights = Path(models[model_key]).resolve(strict=True)
    if not valid(weights, spec):
        raise ValueError('Model verification failed; no replacement download was started')
    params = {'pipeline': PIPELINE, 'source_sha256': sha256(source), 'model_sha256': spec['sha256'],
              'model': model_key, 'mode': mode, 'profile': profile, 'catalog': FINGERPRINT}
    ident = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    job = Path(output_root) / 'jobs' / ident
    job.mkdir(parents=True, exist_ok=True)
    with open(job / '.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This exact separation is already running') from None
        manifest = job / 'result.json'
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result.get('complete') and result.get('parameters') == params:
                if all(sha256(v['path']) == v['sha256'] for v in result['stems'].values()):
                    print('[StemStudio] Reusing completed stems; no GPU work.', flush=True)
                    return result
                raise ValueError('A completed stem changed; preserved for inspection')
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and not allow_cpu:
            raise RuntimeError('Enable Kaggle GPU T4 before running Stem Studio; CPU fallback is disabled')
        if device == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        else:
            torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        progress_path = job / 'progress.json'

        def progress(fields):
            fields.update(device=device, job=ident)
            progress_path.write_text(json.dumps(fields))
            print('[StemStudio] ' + json.dumps(fields), flush=True)

        progress({'stage': 'decode'})
        original, rate = load_audio(source)
        model_rate = spec['config']['audio']['sample_rate']
        mix = resample(original, rate, model_rate)
        if mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)
        # 8-second windows bound T4 memory, batch size one; config/STFT stay intact.
        hop = spec['config']['model']['stft_hop_length']
        chunk_size = (8 * model_rate // hop) * hop
        needed = (mix.shape[-1] * (6 if model_key == 'sw6' else 2) * 2 * 4 * 3
                  + original.nbytes + 512 * 1024**2)
        if shutil.disk_usage(job).free < needed:
            raise RuntimeError('Insufficient output disk for stems and ZIP; original song preserved')
        progress({'stage': 'model_load'})
        model = None
        try:
            model = construct_model(spec, weights).to(device)
            names = (spec['config']['training']['instruments'] if model_key == 'sw6' else ['vocals'])
            estimates = overlap_add(mix, model, len(names), chunk_size, PROFILES[profile], device, progress, interrupt)
            progress({'stage': 'export'})
            files, export = export_stems(original, rate, estimates, model_rate, names, mode, job / 'stems')
            result = {'complete': True, 'parameters': params, 'stems': files, 'source': str(source),
                      'source_frames': original.shape[-1], 'source_rate': rate, 'source_channels': original.shape[0],
                      'duration_seconds': original.shape[-1] / rate, 'device': device,
                      'gpu_name': torch.cuda.get_device_name(0) if device != 'cpu' else None,
                      'peak_vram_bytes': torch.cuda.max_memory_allocated() if device != 'cpu' else 0,
                      'elapsed_seconds': round(time.monotonic() - began, 3),
                      'chunk_samples': chunk_size, 'overlap': PROFILES[profile], 'export': export,
                      'zip_path': str(job / 'all_stems.zip')}
            temporary = job / 'result.writing'
            temporary.write_text(json.dumps(result, indent=2))
            with zipfile.ZipFile(job / 'all_stems.zip.partial', 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name, info in files.items():
                    archive.write(info['path'], name + '.wav')
                archive.writestr('result.json', json.dumps(result, indent=2))
                archive.writestr('README.txt', 'All WAV stems start at 0 and retain the source frame count, sample rate and channels.\nDrag all WAVs into REAPER at 0:00. No individual normalization or lossy encoding was applied.\nSource separation is an estimate; bleed and artifacts may remain.\n')
            (job / 'all_stems.zip.partial').replace(job / 'all_stems.zip')
            temporary.replace(manifest)
            progress({'stage': 'complete', 'elapsed_seconds': result['elapsed_seconds']})
            return result
        except BaseException as exc:
            progress({'stage': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        finally:
            del model
            gc.collect()
            if device != 'cpu':
                torch.cuda.empty_cache()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run_job(**request)
    except Exception as exc:
        Path(sys.argv[2]).write_text(json.dumps({'error': str(exc), 'error_type': type(exc).__name__}))
        raise
    Path(sys.argv[2]).write_text(json.dumps(result))


if __name__ == '__main__':
    main()
