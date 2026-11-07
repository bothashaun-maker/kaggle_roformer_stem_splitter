"""One startup cell → verified attached Dataset → normal ComfyUI → Cloudflare."""
from pathlib import Path
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from stem_studio.assets import DEFAULT_HANDLE, find_pack, sha256, transfer

DATASET_HANDLE = DEFAULT_HANDLE
AUTO_ATTACH_DATASET = True
PORT = 8188
BASE = Path('/kaggle/temp/stem-studio-v1')
OUTPUT = Path('/kaggle/working/stem_studio')
COMFY_REV = '423cbd20511becb1f16122ab518e92e58bfa03d9'
CLOUDFLARE = {'url': 'https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/cloudflared-linux-amd64',
              'bytes': 40274445, 'sha256': '1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872'}
REQUIREMENTS = ['bs-roformer==0.4.0', 'rotary-embedding-torch==0.6.5', 'einops==0.8.2',
                'beartype==0.18.5', 'librosa>=0.10.2,<0.12', 'soundfile>=0.13,<0.14']


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def install_plugin(source, comfy):
    source, comfy = Path(source), Path(comfy)
    plugin = comfy / 'custom_nodes/stem_studio'
    plugin.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('launcher.py', 'open.js'):
            shutil.copyfile(path, plugin / path.name)
    web = plugin / 'web'
    web.mkdir(exist_ok=True)
    shutil.copyfile(source / 'open.js', web / 'open.js')
    graphs = json.loads((source / 'workflows.json').read_text())
    workflow_dir = comfy / 'user/default/workflows'
    workflow_dir.mkdir(parents=True, exist_ok=True)
    for key, graph in graphs.items():
        (web / (key + '.json')).write_text(json.dumps(graph))
        (workflow_dir / ('Stem Studio - ' + key + '.json')).write_text(json.dumps(graph))
    return graphs


def checkout(destination):
    destination = Path(destination)
    if not (destination / '.git').exists():
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise RuntimeError('ComfyUI directory contains unexpected files; preserved')
        run(['git', 'init', destination])
        run(['git', '-C', destination, 'remote', 'add', 'origin', 'https://github.com/Comfy-Org/ComfyUI.git'])
    head = subprocess.run(['git', '-C', str(destination), 'rev-parse', '--verify', 'HEAD'],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if head.returncode:
        remote = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
        if remote != 'https://github.com/Comfy-Org/ComfyUI.git':
            raise RuntimeError('Unexpected ComfyUI remote; existing files preserved')
        run(['git', '-C', destination, 'fetch', '--depth', '1', 'origin', COMFY_REV])
        run(['git', '-C', destination, 'checkout', '--detach', 'FETCH_HEAD'])
    revision = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMFY_REV:
        raise RuntimeError('Unexpected ComfyUI source; existing files preserved: ' + str(destination))


def launch():
    BASE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    children = []
    with open(BASE / 'startup.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Stem Studio is already running; stop its earlier cell') from None
        try:
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', PORT))
                except OSError:
                    raise RuntimeError('Port 8188 is occupied. Stop the earlier music/movie/image startup cell first') from None
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Select Settings → Accelerator → GPU T4 ×2 before running Stem Studio. Model preparation uses None')
            free, total = torch.cuda.mem_get_info(0)
            if free < 8 * 1024**3:
                raise RuntimeError('GPU 0 needs at least 8 GiB free; stop other GPU jobs first')
            print('GPU:', torch.cuda.get_device_name(0), '| CUDA:', torch.version.cuda, '| Torch:', torch.__version__, flush=True)
            print('GPU 0 is used. A second T4 does not pool its memory.', flush=True)
            root, models = find_pack(['/kaggle/input', '/kaggle/working/stem-studio-model-pack'])
            if models is None and AUTO_ATTACH_DATASET:
                try:
                    import kagglehub
                    print('Attaching saved model Dataset:', DATASET_HANDLE, flush=True)
                    attached = kagglehub.dataset_download(DATASET_HANDLE)
                    root, models = find_pack([attached])
                except Exception as exc:
                    raise RuntimeError('Could not attach the saved Stem Studio Dataset. Run Stem_Splitter_Model_Packs.ipynb once, wait for Dataset processing, then Add Input → Your Datasets. No replacement model downloads were started') from exc
            if models is None:
                raise RuntimeError('Attach the verified Stem Studio Dataset or saved model-pack output. No model downloads were started')
            print('Verified reusable model pack:', root, flush=True)
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                if os.geteuid() == 0 and shutil.which('apt-get'):
                    run(['apt-get', '-qq', 'update'])
                    run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
                if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                    raise RuntimeError('FFmpeg/ffprobe is unavailable in this Kaggle image')
            comfy = BASE / 'ComfyUI'
            checkout(comfy)
            python = BASE / 'venv/bin/python'
            if not python.exists():
                run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', python.parent.parent])
            protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy']
            constraints = BASE / 'constraints.txt'
            constraints.write_text('\n'.join(f'{name}=={metadata.version(name)}' for name in protected))
            requirements = BASE / 'requirements.txt'
            requirements.write_text((comfy / 'requirements.txt').read_text() + '\n' + '\n'.join(REQUIREMENTS))
            fingerprint = hashlib.sha256((requirements.read_text() + constraints.read_text() + sys.version).encode()).hexdigest()
            marker = BASE / 'setup.complete'
            if not marker.exists() or marker.read_text() != fingerprint:
                run([sys.executable, '-m', 'pip', '--python', python, 'install', '-r', requirements, '-c', constraints])
                run([python, '-c', 'import torch,bs_roformer,librosa,soundfile,aiohttp; assert torch.cuda.is_available()'])
                marker.write_text(fingerprint)
            source = Path(__file__).parent
            graphs = install_plugin(source, comfy)
            cloudflare = BASE / 'cloudflared'
            transfer(CLOUDFLARE, cloudflare)
            cloudflare.chmod(0o700)
            env = dict(os.environ, STEM_STUDIO_OUTPUT=str(OUTPUT), STEM_STUDIO_MODELS=json.dumps(models))
            def spawn(args, name):
                with open(OUTPUT / name, 'w') as log:
                    child = subprocess.Popen(list(map(str, args)), stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, env=env, cwd=comfy)
                children.append(child)
                return child
            server = spawn([python, comfy / 'main.py', '--listen', '127.0.0.1', '--port', PORT,
                            '--preview-method', 'none', '--cache-none', '--disable-auto-launch',
                            '--output-directory', OUTPUT, '--max-upload-size', '4096'], 'comfy.log')
            deadline = time.monotonic() + 240
            info = None
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('ComfyUI stopped. See /kaggle/working/stem_studio/comfy.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/object_info', timeout=3) as response:
                        info = json.load(response)
                    break
                except OSError:
                    time.sleep(1)
            if info is None:
                raise RuntimeError('ComfyUI startup timed out; see comfy.log')
            for graph in graphs.values():
                for node in graph['nodes']:
                    if node['type'] not in info:
                        raise RuntimeError('Required node unavailable: ' + node['type'] + '; see comfy.log')
            provenance = {'comfy': COMFY_REV, 'dataset': DATASET_HANDLE, 'models': models,
                          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
            (OUTPUT / 'startup.json').write_text(json.dumps(provenance, indent=2))
            with open(OUTPUT / 'packages.txt', 'w') as stream:
                run([sys.executable, '-m', 'pip', '--python', python, 'freeze'], stdout=stream)
            tunnel = spawn([cloudflare, 'tunnel', '--url', f'http://127.0.0.1:{PORT}', '--no-autoupdate'], 'tunnel.log')
            deadline = time.monotonic() + 90
            url = None
            while time.monotonic() < deadline:
                matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', (OUTPUT / 'tunnel.log').read_text())
                if matches:
                    url = matches[-1]
                    break
                if tunnel.poll() is not None:
                    raise RuntimeError('Cloudflare stopped; see tunnel.log')
                time.sleep(1)
            if url is None:
                raise RuntimeError('Cloudflare did not provide a URL; see tunnel.log')
            print('\nSTEM STUDIO IS READY', flush=True)
            print('Six stems:', url + '/?stem_studio=six', flush=True)
            print('Four stems:', url + '/?stem_studio=four', flush=True)
            print('Vocals / instrumental:', url + '/?stem_studio=vocals', flush=True)
            print('UPLOAD YOUR WAV:', url + '/stem/upload', flush=True)
            print('Upload → select song → press Run once. Balanced is overlap 2; Quality is overlap 4 (slower).', flush=True)
            print('Download ZIP on the Export node. Files also appear under /kaggle/working/stem_studio/jobs/.', flush=True)
            print('Keep this startup cell running. Anyone with this temporary URL can access the studio.', flush=True)
            while server.poll() is None and tunnel.poll() is None:
                time.sleep(2)
            raise RuntimeError('ComfyUI or the tunnel stopped; inspect the saved logs')
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()


if __name__ == '__main__':
    launch()
