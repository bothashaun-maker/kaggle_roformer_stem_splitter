"""Visible source → settings → separation → preview/export ComfyUI nodes."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import folder_paths
from aiohttp import web
from server import PromptServer
from .assets import CATALOG
from .upload import UploadStore, CHUNK_BYTES, EXTENSIONS

ROOT = Path(os.environ.get('STEM_STUDIO_OUTPUT', '/kaggle/working/stem_studio'))
INPUT = ROOT / 'input'
NO_SONG = '(upload a song first)'
LABELS = {m['label']: key for key, m in CATALOG.items()}


def songs():
    INPUT.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in INPUT.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def source_path(audio, attached_path):
    if attached_path.strip():
        path = Path(attached_path.strip()).expanduser().resolve(strict=True)
        if not path.is_file() or not any(path.is_relative_to(r) for r in (Path('/kaggle/input'), ROOT.resolve())):
            raise ValueError('Use a file under /kaggle/input or the Stem Studio folder')
        return path
    if audio not in songs():
        raise ValueError('Upload a song using the upload link and select it from the list')
    return INPUT / audio


class SSLoadSong:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': (songs() or [NO_SONG],),
                'attached_path': ('STRING', {'default': '', 'tooltip': 'Optional /kaggle/input path; otherwise use the uploaded song list.'})}}
    RETURN_TYPES = ('SS_SOURCE',)
    RETURN_NAMES = ('song',)
    FUNCTION = 'load'
    CATEGORY = 'Stem Studio'

    @classmethod
    def VALIDATE_INPUTS(cls, audio, attached_path):
        try:
            source_path(audio, attached_path)
            return True
        except (ValueError, OSError) as exc:
            return str(exc)

    @classmethod
    def IS_CHANGED(cls, audio, attached_path):
        path = source_path(audio, attached_path)
        return str(path) + str(path.stat().st_mtime_ns) + str(path.stat().st_size)

    def load(self, audio, attached_path):
        return (str(source_path(audio, attached_path)),)


class SSSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': (list(LABELS),),
                'output_mode': (['Six stems', 'Four stems', 'Vocals + instrumental'],),
                'quality': (['Balanced', 'Quality', 'Maximum overlap (slower)'],)}}
    RETURN_TYPES = ('SS_SETTINGS',)
    RETURN_NAMES = ('settings',)
    FUNCTION = 'configure'
    CATEGORY = 'Stem Studio'

    def configure(self, model, output_mode, quality):
        if (LABELS[model] == 'sw6') != (output_mode in ('Four stems', 'Six stems')):
            raise ValueError('For four/six stems choose BS RoFormer SW. For a vocal model choose Vocals + instrumental')
        return ({'model_key': LABELS[model], 'mode': output_mode, 'profile': quality},)


class SSSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'song': ('SS_SOURCE',), 'settings': ('SS_SETTINGS',)}}
    RETURN_TYPES = ('SS_STEMS',)
    RETURN_NAMES = ('stems',)
    FUNCTION = 'separate'
    CATEGORY = 'Stem Studio'

    def separate(self, song, settings):
        import comfy.model_management as mm
        ROOT.mkdir(parents=True, exist_ok=True)
        request = dict(source=song, models=json.loads(os.environ['STEM_STUDIO_MODELS']),
                       output_root=str(ROOT), **settings)
        ident = uuid.uuid4().hex
        command = ROOT / ('request-' + ident + '.json')
        answer = ROOT / ('answer-' + ident + '.json')
        command.write_text(json.dumps(request))
        # Package name is private to this plugin, independent of ComfyUI's import name.
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        worker = subprocess.Popen([sys.executable, '-m', Path(__file__).parent.name + '.engine', str(command), str(answer)], env=env, start_new_session=True)
        try:
            while worker.poll() is None:
                mm.throw_exception_if_processing_interrupted()
                time.sleep(.25)
            if worker.returncode:
                error = json.loads(answer.read_text()) if answer.exists() else {}
                detail = str(error.get('error', 'Worker exited without a result'))
                raise RuntimeError('Separation stopped: ' + detail + '\nSee /kaggle/working/stem_studio/comfy.log and jobs/*/progress.json. Saved stages are preserved.')
            return (json.loads(answer.read_text()),)
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
            command.unlink(missing_ok=True)
            answer.unlink(missing_ok=True)


class SSPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'stems': ('SS_STEMS',),
                'stem': (['vocals', 'instrumental', 'drums', 'bass', 'guitar', 'piano', 'other'],),
                'start_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 7200.}),
                'length_seconds': ('FLOAT', {'default': 20., 'min': 1., 'max': 60.})}}
    RETURN_TYPES = ('AUDIO',)
    FUNCTION = 'preview'
    CATEGORY = 'Stem Studio'

    def preview(self, stems, stem, start_seconds, length_seconds):
        import soundfile as sf
        import torch
        if stem not in stems['stems']:
            raise ValueError('This job contains: ' + ', '.join(stems['stems']))
        record = stems['stems'][stem]
        start = int(start_seconds * record['sample_rate'])
        if start >= record['frames']:
            raise ValueError('Preview starts after the song ends')
        data, rate = sf.read(record['path'], start=start, frames=int(length_seconds * record['sample_rate']),
                             dtype='float32', always_2d=True)
        return ({'waveform': torch.from_numpy(data.T.copy()).unsqueeze(0), 'sample_rate': rate},)


class SSExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'stems': ('SS_STEMS',)}}
    RETURN_TYPES = ()
    FUNCTION = 'export'
    OUTPUT_NODE = True
    CATEGORY = 'Stem Studio'

    def export(self, stems):
        job = Path(stems['zip_path']).parent.name
        text = (f"Saved {len(stems['stems'])} stems. {stems['elapsed_seconds']:.1f} seconds on {stems['device']}.\n"
                f"{stems['source_rate']} Hz; {stems['source_frames']} frames; float WAV.\n"
                f"Download ZIP or find files under {Path(stems['zip_path']).parent}")
        return {'ui': {'text': [text], 'stem_download': [job]}, 'result': ()}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SSLoadSong, SSSettings, SSSeparate, SSPreview, SSExport)}
NODE_DISPLAY_NAME_MAPPINGS = {'SSLoadSong': 'Your song', 'SSSettings': 'Model & quality',
                            'SSSeparate': 'Split into stems', 'SSPreview': 'Audition a stem', 'SSExport': 'Download all stems'}
WEB_DIRECTORY = 'web'


def register_routes():
    routes = PromptServer.instance.routes
    store = UploadStore(INPUT)

    @routes.get('/stem/upload')
    async def page(request):
        return web.Response(text=Path(__file__).with_name('uploader.html').read_text(), content_type='text/html')

    @routes.get('/stem/songs')
    async def song_list(request):
        return web.json_response(songs())

    @routes.post('/stem/uploads')
    async def begin(request):
        try:
            data = await request.json()
            return web.json_response(store.begin(data['name'], data['size'], data.get('id')))
        except (ValueError, KeyError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @routes.put('/stem/uploads/{ident}/{offset}')
    async def piece(request):
        try:
            data = await request.content.read(CHUNK_BYTES + 1)
            # read() may return early on a live HTTP stream; gather the bounded body.
            while len(data) <= CHUNK_BYTES and not request.content.at_eof():
                data += await request.content.read(CHUNK_BYTES + 1 - len(data))
            return web.json_response(store.append(request.match_info['ident'], int(request.match_info['offset']), data))
        except (ValueError, KeyError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @routes.post('/stem/uploads/{ident}/finish')
    async def finish(request):
        try:
            return web.json_response(store.finish(request.match_info['ident']))
        except (ValueError, KeyError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @routes.get('/stem/download/{ident}')
    async def download(request):
        ident = request.match_info['ident']
        if len(ident) != 20 or any(c not in '0123456789abcdef' for c in ident):
            raise web.HTTPNotFound()
        path = ROOT / 'jobs' / ident / 'all_stems.zip'
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Content-Disposition': 'attachment; filename="all_stems.zip"'})


register_routes()
