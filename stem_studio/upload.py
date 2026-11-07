"""Resumable 4-MiB uploads. No original-audio conversion or large HTTP request."""
from pathlib import Path
import json
import re
import shutil
import uuid

EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.aiff', '.aif'}
CHUNK_BYTES = 4 * 1024**2
MAX_BYTES = 5 * 1024**3


class UploadStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / '.partial').mkdir(exist_ok=True)

    def paths(self, ident):
        if not re.fullmatch(r'[a-f0-9]{32}', ident):
            raise ValueError('Invalid upload ID')
        base = self.root / '.partial' / ident
        return base.with_suffix('.json'), base.with_suffix('.part')

    def begin(self, name, size, ident=None):
        if not isinstance(name, str) or not name or len(name) > 180 or any(x in name for x in ('/', '\\', '\x00')):
            raise ValueError('Invalid filename')
        if Path(name).suffix.lower() not in EXTENSIONS:
            raise ValueError('Upload WAV, FLAC, MP3, M4A, OGG, OPUS or AIFF')
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_BYTES:
            raise ValueError('File size must be between 1 byte and 5 GiB')
        ident = ident or uuid.uuid4().hex
        meta, part = self.paths(ident)
        if meta.exists():
            data = json.loads(meta.read_text())
            if data['name'] != name or data['size'] != size:
                raise ValueError('Upload metadata differs from the saved upload')
            if data.get('complete'):
                return dict(data, offset=size)
        else:
            data = {'id': ident, 'name': name, 'size': size}
            meta.write_text(json.dumps(data))
        offset = part.stat().st_size if part.exists() else 0
        if shutil.disk_usage(self.root).free < size - offset + 256 * 1024**2:
            raise ValueError('Insufficient server disk for this upload')
        return dict(data, offset=offset)

    def append(self, ident, offset, payload):
        import fcntl
        meta, part = self.paths(ident)
        data = json.loads(meta.read_text())
        if data.get('complete'):
            raise ValueError('Upload is already complete')
        if not 0 < len(payload) <= CHUNK_BYTES:
            raise ValueError('Invalid upload chunk size')
        with open(part, 'a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0, 2)
            if stream.tell() != offset:
                raise ValueError('Chunk offset mismatch; resume from saved offset')
            if offset + len(payload) > data['size']:
                raise ValueError('Chunk exceeds declared file size')
            stream.write(payload)
            return {'offset': offset + len(payload)}

    def finish(self, ident):
        meta, part = self.paths(ident)
        data = json.loads(meta.read_text())
        if data.get('complete'):
            return data
        if not part.is_file() or part.stat().st_size != data['size']:
            raise ValueError('Upload is incomplete; resume before finishing')
        filename = ident[:10] + '_' + data['name']
        destination = self.root / filename
        if destination.exists():
            raise ValueError('Destination already exists; preserved')
        part.replace(destination)
        data.update(complete=True, filename=filename, path=str(destination))
        meta.write_text(json.dumps(data))
        return data
