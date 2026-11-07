"""Verified, CPU-only model packs. Rendering never downloads replacement weights."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request

CATALOG = json.loads(Path(__file__).with_name('catalog.json').read_text())
FINGERPRINT = hashlib.sha256(json.dumps(CATALOG, sort_keys=True).encode()).hexdigest()[:10]
DEFAULT_HANDLE = 'janpierrrewiet/stem-studio-assets-v1-' + FINGERPRINT


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def valid(path, spec):
    path = Path(path)
    return path.is_file() and path.stat().st_size == spec['bytes'] and sha256(path) == spec['sha256']


def transfer(spec, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if valid(target, spec):
            print('Verified; reused:', target.name, flush=True)
            return
        raise RuntimeError('Existing model failed verification; preserved: ' + str(target))
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == spec['bytes'] and valid(partial, spec):
            partial.replace(target)
            return
        if offset >= spec['bytes']:
            raise RuntimeError('Invalid partial model; preserved: ' + str(partial))
        if shutil.disk_usage(target.parent).free < spec['bytes'] - offset + 512 * 1024**2:
            raise RuntimeError('Insufficient free disk for ' + target.name)
        request = urllib.request.Request(spec['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected download range; partial model preserved')
                written = offset if resume else 0
                reported = written
                with open(partial, 'ab' if resume else 'wb') as stream:
                    while block := response.read(8 * 1024**2):
                        stream.write(block)
                        written += len(block)
                        if written - reported >= 128 * 1024**2:
                            print(f'{target.name}: {written / spec["bytes"]:.0%}', flush=True)
                            reported = written
            if not valid(partial, spec):
                raise RuntimeError('Model size/checksum mismatch; partial model preserved')
            partial.replace(target)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def verify_pack(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not isinstance(manifest, dict) or manifest.get('schema') != 1 or manifest.get('fingerprint') != FINGERPRINT or not manifest.get('complete'):
        raise RuntimeError('This model pack is incomplete or belongs to a different Stem Studio version')
    paths = {}
    for key, spec in CATALOG.items():
        path = root / spec['name']
        if not valid(path, spec):
            raise RuntimeError('Model size/checksum mismatch: ' + str(path))
        paths[key] = str(path.resolve())
    return paths


def find_pack(roots):
    for root in map(Path, roots):
        if not root.exists():
            continue
        candidates = [root / 'manifest.json'] + list(root.rglob('manifest.json'))
        for marker in dict.fromkeys(candidates):
            try:
                data = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get('fingerprint') == FINGERPRINT:
                return marker.parent, verify_pack(marker.parent)
    return None, None


def build_pack(root, input_root='/kaggle/input'):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / '.builder.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This pack is already being built') from None
        manifest = root / 'manifest.json'
        if manifest.exists():
            manifest.replace(root / 'manifest.previous.json')
        for spec in CATALOG.values():
            target = root / spec['name']
            if not target.exists() and Path(input_root).exists():
                for candidate in Path(input_root).rglob(spec['name']):
                    if valid(candidate, spec):
                        if shutil.disk_usage(root).free < spec['bytes'] + 512 * 1024**2:
                            raise RuntimeError('Insufficient disk to copy attached weights')
                        temp = target.with_suffix('.copying')
                        shutil.copyfile(candidate, temp)
                        temp.replace(target)
                        break
            transfer(spec, target)
            if target.is_symlink():
                raise RuntimeError('A saved pack must contain actual files, not input symlinks')
        data = {'schema': 1, 'fingerprint': FINGERPRINT, 'complete': True, 'files': CATALOG}
        temp = root / 'manifest.writing'
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(manifest)
        verify_pack(root)
        print('PACK COMPLETE:', root, flush=True)
        return root


def publish_dataset(root, handle=DEFAULT_HANDLE, submit=None):
    """Create a private named Dataset only; never update an existing Dataset."""
    root = Path(root)
    verify_pack(root)
    allowed = {spec['name'] for spec in CATALOG.values()} | {'manifest.json', 'manifest.previous.json', '.builder.lock'}
    if any(path.name not in allowed or not path.is_file() for path in root.iterdir()):
        raise RuntimeError('Unexpected files in the model-pack folder; no Dataset upload started')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', handle):
        raise ValueError('DATASET_HANDLE must be owner/dataset-name')
    receipt = root.parent / (root.name + '-dataset-receipt.json')
    prior = json.loads(receipt.read_text()) if receipt.exists() else {}
    if prior.get('handle') == handle and prior.get('fingerprint') == FINGERPRINT:
        print('Previously submitted:', 'https://www.kaggle.com/datasets/' + handle, flush=True)
        return handle
    if submit is None:
        # Same KaggleHub authentication as the user's successful music-asset upload.
        # Explicit create-only request prevents versioning an unrelated/public Dataset.
        if not os.environ.get('KAGGLE_API_TOKEN'):
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
            except Exception:
                pass  # Kaggle's ambient authentication may already be sufficient.
        from kagglehub.clients import build_kaggle_client
        from kagglehub.gcs_upload import upload_files_and_directories
        from kagglesdk.blobs.types.blob_api_service import ApiBlobType
        from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest

        def submit(folder, owner, slug, private):
            upload = upload_files_and_directories(str(folder), item_type=ApiBlobType.DATASET,
                ignore_patterns=['.*', '*.partial', '*.previous.json'])
            proto = upload.to_proto()
            request = ApiCreateDatasetRequest()
            request.owner_slug, request.slug = owner, slug
            request.title = 'Stem Studio Models ' + FINGERPRINT
            request.files, request.directories = proto.files, proto.directories
            request.is_private = private
            with build_kaggle_client() as client:
                client.datasets.dataset_api_client.create_dataset(request)
    owner, slug = handle.split('/')
    print('Creating PRIVATE Dataset:', 'https://www.kaggle.com/datasets/' + handle, flush=True)
    submit(root, owner, slug, True)
    receipt.write_text(json.dumps({'handle': handle, 'submitted_private': True, 'fingerprint': FINGERPRINT}, indent=2))
    print('PRIVATE DATASET SUBMITTED. Wait for Kaggle processing:', 'https://www.kaggle.com/datasets/' + handle, flush=True)
    return handle
