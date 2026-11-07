"""Build self-contained Kaggle notebooks and ordinary connected ComfyUI canvases."""
from pathlib import Path
import ast
import base64
import json
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'stem_studio'


def workflow(kind):
    catalog = json.loads((SOURCE / 'catalog.json').read_text())
    nodes, links = [], []

    def add(node_type, position, widgets, inputs, outputs, title, size=(360, 220)):
        ident = len(nodes) + 1
        nodes.append({'id': ident, 'type': node_type, 'pos': list(position), 'size': list(size),
            'flags': {}, 'order': ident - 1, 'mode': 0, 'title': title,
            'inputs': [{'name': name, 'type': typ, 'link': None} for name, typ in inputs],
            'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in outputs],
            'widgets_values': widgets, 'properties': {'Node name for S&R': node_type}})
        return ident

    def connect(source, output, target, input_slot):
        ident = len(links) + 1
        typ = nodes[source - 1]['outputs'][output]['type']
        links.append([ident, source, output, target, input_slot, typ])
        nodes[source - 1]['outputs'][output]['links'].append(ident)
        nodes[target - 1]['inputs'][input_slot]['link'] = ident

    mode = {'six': 'Six stems', 'four': 'Four stems', 'vocals': 'Vocals + instrumental'}[kind]
    model = catalog['kim' if kind == 'vocals' else 'sw6']['label']
    song = add('SSLoadSong', (0, 0), ['(upload a song first)', ''], [], [('song', 'SS_SOURCE')], '1 · Upload / select your original song')
    settings = add('SSSettings', (0, 300), [model, mode, 'Balanced'], [], [('settings', 'SS_SETTINGS')], '2 · Model and quality — one window at a time')
    separate = add('SSSeparate', (430, 0), [], [('song', 'SS_SOURCE'), ('settings', 'SS_SETTINGS')], [('stems', 'SS_STEMS')], '3 · Split song on GPU')
    export = add('SSExport', (860, 0), [], [('stems', 'SS_STEMS')], [], '4 · Download aligned WAV stems + ZIP', (410, 260))
    audition = add('SSPreview', (430, 300), ['vocals', 0., 20.], [('stems', 'SS_STEMS')], [('AUDIO', 'AUDIO')], '5 · Choose a stem and audition position')
    preview = add('PreviewAudio', (860, 340), [], [('audio', 'AUDIO')], [], '6 · Listen here; original export remains float WAV')
    connect(song, 0, separate, 0)
    connect(settings, 0, separate, 1)
    connect(separate, 0, export, 0)
    connect(separate, 0, audition, 0)
    connect(audition, 0, preview, 0)
    return {'last_node_id': len(nodes), 'last_link_id': len(links), 'nodes': nodes, 'links': links,
            'groups': [], 'config': {}, 'extra': {'ds': {'scale': .8, 'offset': [150, 160]}}, 'version': .4}


def notebook(title, markdown, code):
    ast.parse(code)
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'},
                         'kaggle': {'language': 'python', 'enableInternet': True}},
            'cells': [{'id': 'instructions', 'cell_type': 'markdown', 'metadata': {}, 'source': markdown.splitlines(True)},
                      {'id': 'startup', 'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(True),
                       'execution_count': None, 'outputs': []}]}


def payload_code(files):
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    return '''import base64, json, sys, zlib
from pathlib import Path
SOURCE_ROOT = Path('/kaggle/temp/stem-studio-source-v1')
for name in list(sys.modules):
    if name == 'stem_studio' or name.startswith('stem_studio.'):
        del sys.modules[name]
for relative, text in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
    path = SOURCE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0, str(SOURCE_ROOT))
'''.replace('PAYLOAD', repr(payload))


def build():
    graphs = {key: workflow(key) for key in ('six', 'four', 'vocals')}
    (SOURCE / 'workflows.json').write_text(json.dumps(graphs, indent=2) + '\n')
    files = {'stem_studio/' + p.name: p.read_text() for p in SOURCE.iterdir()
             if p.is_file() and p.suffix in ('.py', '.json', '.js', '.html')}
    for path, content in files.items():
        if path.endswith('.py'):
            ast.parse(content)
    code = payload_code(files) + '''
from stem_studio.launcher import launch
launch()
'''
    intro = '''# Stem Studio — normal ComfyUI canvas on Kaggle

**One startup cell → Cloudflare link → upload WAV → Run → download stems.**

First time: run **Stem_Splitter_Model_Packs.ipynb** with **Accelerator: None**.
It prepares about **2.1 GiB** of verified models and submits a private named Kaggle Dataset.
Wait for Dataset processing. This notebook automatically tries to attach that saved Dataset;
you can also use **Add Input → Your Datasets** or saved builder output. No model weights
are downloaded from Hugging Face during startup or separation.

Every music session: **GPU T4 ×2**, **Internet ON**, run the single cell below and leave it running.
GPU 0 performs separation; no silent CPU fallback. No separation starts automatically.
The three canvases also appear in ComfyUI's Workflows menu.

- **Vocals + instrumental:** Kim Mel-Band RoFormer; Viperx BS-RoFormer is an alternative.
- **Six stems:** BS-RoFormer SW → vocals, drums, bass, guitar, piano, other.
- **Four stems:** the same six-stem model, with guitar + piano added to other.

Balanced uses 8-second windows, batch size one and overlap 2. Quality uses overlap 4;
Maximum uses overlap 8 and takes much longer. More overlap is not guaranteed to sound better.
Model weights are loaded in an independent job process and released when it exits.

Use the **Upload song** button or printed upload link for large WAVs. Uploads use resumable
4-MiB pieces; no conversion is required. Select the uploaded song, choose a model and press Run once.
Audition any saved stem. **Download all stems ZIP** appears on the Export node after completion.
All stems are **32-bit float WAV** at the original sample rate and exact frame count, starting at zero.
Inference resamples internally to 44.1 kHz; this does not restore information lost by the model.
Stems are not independently normalized. Guitar/piano separation and growled vocals can still bleed.

Files and progress logs: `/kaggle/working/stem_studio/`. Download the ZIP or save a notebook
version with output files before ending the session. Saving model assets does not save your songs.
Software still installs over Kaggle's connection; the model Dataset avoids repeated weight downloads.

Local model/CPU and browser checks are documented in README.md. T4 runtime, full-song separation
quality and authenticated Kaggle Dataset submission require live Kaggle validation.
'''
    studio = notebook('Stem Studio', intro, code)
    for name in ('Stem_Splitter_ComfyUI_Kaggle.ipynb', 'kaggle_roformer_stem_splitter.ipynb'):
        (ROOT / name).write_text(json.dumps(studio, indent=1, ensure_ascii=False) + '\n')
    builder_files = {key: value for key, value in files.items()
                     if key in ('stem_studio/__init__.py', 'stem_studio/assets.py', 'stem_studio/catalog.json')}
    builder_code = payload_code(builder_files) + '''
# Run once with Accelerator: None and Internet ON. No GPU/model inference is used.
AUTO_CREATE_DATASET = True
from stem_studio.assets import DEFAULT_HANDLE, build_pack, publish_dataset
DATASET_HANDLE = DEFAULT_HANDLE  # Your Kaggle account: janpierrrewiet
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub==1.0.3'], check=True)
pack = build_pack('/kaggle/working/stem-studio-model-pack')
if AUTO_CREATE_DATASET:
    try:
        publish_dataset(pack, DATASET_HANDLE)
    except Exception as error:
        print('Dataset submission did not complete. Verified models remain in', pack)
        print('If the named Dataset already exists, attach it; this builder never updates existing Datasets.')
        print('For authentication failure, enable Kaggle Secret KAGGLE_API_TOKEN, then rerun this cell; existing verified models are reused.')
        print('You can also Save Version with outputs and attach that saved pack. No PC transfer is required.')
        raise RuntimeError('Private Dataset submission failed; models preserved. See the preceding Kaggle error.') from error
print('SETUP COMPLETE. Open the Dataset link above, wait for processing, then use Stem_Splitter_ComfyUI_Kaggle.ipynb with T4 GPUs.')
'''
    builder_intro = '''# Stem Studio — save models once as a private named Dataset

1. Select **Accelerator: None** and **Internet ON**.
2. Run the single cell. It downloads and verifies approximately **2.1 GiB** on Kaggle.
3. Wait for **PRIVATE DATASET SUBMITTED**, open its link and wait for processing.
4. Open **Stem_Splitter_ComfyUI_Kaggle.ipynb**, select T4 ×2 and run its startup cell.

The Dataset is named **Stem Studio Models** followed by its version fingerprint, under your
Kaggle account `janpierrrewiet`. You can find and attach it whenever you return.
No model download or upload passes through your PC. This does not package songs or generated stems.

Automatic creation uses KaggleHub's Kaggle authentication and explicitly requests **private**
visibility. It never updates an existing Dataset. If authentication is unavailable, enable the
private Kaggle Secret **KAGGLE_API_TOKEN** for this notebook; never paste tokens into code or chat.
Verified downloads are retained if submission fails. Saving a notebook version **with outputs**
also preserves a reusable pack; a draft alone does not.

Model sources and SHA256 values are in `manifest.json`; original model licenses apply.
Dataset submission/processing and auto-attachment have not been tested with your Kaggle account here.
'''
    (ROOT / 'Stem_Splitter_Model_Packs.ipynb').write_text(json.dumps(notebook('Stem model packs', builder_intro, builder_code), indent=1, ensure_ascii=False) + '\n')
    print('Built Stem_Splitter_ComfyUI_Kaggle.ipynb, Stem_Splitter_Model_Packs.ipynb and legacy URL alias')


if __name__ == '__main__':
    build()
