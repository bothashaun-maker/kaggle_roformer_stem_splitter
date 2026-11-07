"""Checks for model-pack integrity, uploads and full-length sample alignment."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from stem_studio import assets, engine
from stem_studio.upload import UploadStore, CHUNK_BYTES


class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = {}
        for key in ('one', 'two'):
            data = (key * 50).encode()
            path = self.root / (key + '.source')
            path.write_bytes(data)
            self.catalog[key] = {'name': key + '.ckpt', 'url': path.as_uri(),
                                 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        self.patch = patch.object(assets, 'CATALOG', self.catalog)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_pack_download_reuse_and_corruption_preservation(self):
        root = assets.build_pack(self.root / 'pack', self.root / 'none')
        self.assertEqual(set(assets.verify_pack(root)), set(self.catalog))
        with patch.object(assets.urllib.request, 'urlopen', side_effect=AssertionError('No repeated download')):
            assets.build_pack(root, self.root / 'none')
        target = root / 'one.ckpt'
        target.write_bytes(b'broken')
        with self.assertRaisesRegex(RuntimeError, 'preserved'):
            assets.build_pack(root, self.root / 'none')
        self.assertEqual(target.read_bytes(), b'broken')
        self.assertFalse((root / 'manifest.json').exists())

    def test_input_files_are_copied_into_durable_pack(self):
        attached = self.root / 'input'
        attached.mkdir()
        for spec in self.catalog.values():
            (attached / spec['name']).write_bytes(Path(spec['url'][7:]).read_bytes())
        with patch.object(assets.urllib.request, 'urlopen', side_effect=AssertionError('Use attached weights')):
            root = assets.build_pack(self.root / 'pack', attached)
        self.assertTrue(all(not Path(p).is_symlink() for p in assets.verify_pack(root).values()))

    def test_dataset_submission_is_private_create_only_and_idempotent(self):
        root = assets.build_pack(self.root / 'pack', self.root / 'none')
        calls = []
        def submit(folder, owner, slug, private):
            calls.append((owner, slug, private))
        assets.publish_dataset(root, 'tester/stem-models', submit)
        assets.publish_dataset(root, 'tester/stem-models', submit)
        self.assertEqual(calls, [('tester', 'stem-models', True)])

    def test_failed_submission_retains_verified_pack_without_success_receipt(self):
        root = assets.build_pack(self.root / 'pack', self.root / 'none')
        with self.assertRaisesRegex(RuntimeError, 'Authentication'):
            assets.publish_dataset(root, 'tester/stem-models', lambda *a: (_ for _ in ()).throw(RuntimeError('Authentication')))
        self.assertEqual(len(assets.verify_pack(root)), 2)
        self.assertFalse((root.parent / 'pack-dataset-receipt.json').exists())

    def test_real_sdk_request_shape_and_upload_signature_without_network(self):
        from types import SimpleNamespace
        from kagglehub.gcs_upload import upload_files_and_directories
        root = assets.build_pack(self.root / 'pack', self.root / 'none')
        with patch('kagglehub.gcs_upload.upload_files_and_directories', autospec=True) as upload, \
             patch('kagglehub.clients.build_kaggle_client') as client:
            upload.return_value.to_proto.return_value = SimpleNamespace(files=[], directories=[])
            assets.publish_dataset(root, 'tester/stem-sdk')
            request = client.return_value.__enter__.return_value.datasets.dataset_api_client.create_dataset.call_args.args[0]
            self.assertIs(request.is_private, True)
            self.assertEqual((request.owner_slug, request.slug), ('tester', 'stem-sdk'))
            self.assertEqual(upload.call_count, 1)

    def test_wrong_manifest_never_passes_as_attached_models(self):
        root = assets.build_pack(self.root / 'pack', self.root / 'none')
        m = root / 'manifest.json'
        d = json.loads(m.read_text());d['complete'] = False;m.write_text(json.dumps(d))
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            assets.verify_pack(root)


class UploadTests(unittest.TestCase):
    def test_interrupted_upload_resumes_at_exact_offset_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            store = UploadStore(temp)
            data = b'WAVE' * (CHUNK_BYTES // 4 + 11)
            info = store.begin('Original Mix.wav', len(data))
            ident = info['id']
            self.assertEqual(store.append(ident, 0, data[:CHUNK_BYTES])['offset'], CHUNK_BYTES)
            self.assertEqual(store.begin('Original Mix.wav', len(data), ident)['offset'], CHUNK_BYTES)
            with self.assertRaisesRegex(ValueError, 'offset'):
                store.append(ident, 0, b'duplicate')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                store.finish(ident)
            store.append(ident, CHUNK_BYTES, data[CHUNK_BYTES:])
            done = store.finish(ident)
            self.assertEqual(Path(done['path']).read_bytes(), data)
            self.assertEqual(store.finish(ident), done)

    def test_upload_rejects_path_escape_size_conflicts_and_oversized_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            store = UploadStore(temp)
            for name,size in [('../song.wav', 10), ('C:\\song.wav', 10), ('song.exe', 10),
                              ('song.wav', True), ('song.wav', 0), ('song.wav', 6 * 1024**3)]:
                with self.subTest(name=name, size=size), self.assertRaises(ValueError):
                    store.begin(name, size)
            i = store.begin('song.wav', 10)['id']
            with self.assertRaisesRegex(ValueError, 'differs'):
                store.begin('another.wav', 10, i)
            with self.assertRaisesRegex(ValueError, 'size'):
                store.append(i, 0, b'x' * (CHUNK_BYTES + 1))
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                store.append(i, 0, b'x' * 11)


class SignalTests(unittest.TestCase):
    def test_overlap_add_preserves_short_long_and_final_partial_samples(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x.unsqueeze(1)
        rng = np.random.default_rng(1)
        for length in (1, 4095, 8193, 480127):
            audio = rng.normal(0, .05, (2, length)).astype(np.float32)
            for overlap in (2, 4, 8):
                with self.subTest(length=length, overlap=overlap):
                    result = engine.overlap_add(audio, Identity(), 1, 8192, overlap, 'cpu')
                    self.assertEqual(result.shape, (1, 2, length))
                    np.testing.assert_allclose(result[0], audio, atol=1e-7, rtol=1e-6)

    def test_interruption_is_checked_between_chunks(self):
        def stop():
            raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):
            engine.overlap_add(np.ones((2, 9000), np.float32), torch.nn.Identity(), 1, 8192, 2, 'cpu', interrupt=stop)

    def test_float_export_preserves_source_clock_and_complement_without_normalization(self):
        rng = np.random.default_rng(2)
        for channels, rate in ((1, 48000), (2, 44100)):
            with self.subTest(channels=channels, rate=rate), tempfile.TemporaryDirectory() as temp:
                original = rng.normal(0, .4, (channels, 48017)).astype(np.float32)
                source = np.repeat(original, 2, axis=0) if channels == 1 else original
                estimates = engine.resample(source * .2, rate, 44100)[None]
                files, diagnostic = engine.export_stems(original, rate, estimates, 44100, ['vocals'], 'Vocals + instrumental', temp)
                self.assertEqual(set(files), {'vocals', 'instrumental'})
                combined = 0
                for record in files.values():
                    data, sr = sf.read(record['path'], dtype='float32', always_2d=True)
                    self.assertEqual((len(data), sr, data.shape[1]), (original.shape[-1], rate, channels))
                    self.assertEqual(sf.info(record['path']).subtype, 'FLOAT')
                    combined = combined + data.T
                np.testing.assert_allclose(combined, original, atol=1e-6)
                self.assertLess(diagnostic['sum_residual_rms'], 1e-6)

    def test_four_stem_mode_combines_instruments_without_dropping_samples(self):
        names = ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
        estimates = np.stack([np.full((2, 4411), i * .1, np.float32) for i in range(6)])
        with tempfile.TemporaryDirectory() as temp:
            files, _ = engine.export_stems(np.ones((2, 4411), np.float32), 44100, estimates, 44100, names, 'Four stems', temp)
            self.assertEqual(set(files), {'vocals', 'drums', 'bass', 'other'})
            data, _ = sf.read(files['other']['path'])
            np.testing.assert_allclose(data, 1.1, atol=1e-6)

    def test_invalid_model_mode_never_starts_inference(self):
        with self.assertRaisesRegex(ValueError, 'Choose a vocal model'):
            engine.run_job('unused', 'kim', 'Six stems', 'Balanced', {}, 'unused')

    def test_worker_preserves_the_actual_failure_for_the_canvas(self):
        with tempfile.TemporaryDirectory() as temp:
            request, answer = Path(temp) / 'request.json', Path(temp) / 'answer.json'
            request.write_text(json.dumps(dict(source='unused', model_key='missing', mode='Six stems',
                                              profile='Balanced', models={}, output_root=temp)))
            with patch('sys.argv', ['engine.py', str(request), str(answer)]), self.assertRaises(ValueError):
                engine.main()
            result = json.loads(answer.read_text())
            self.assertEqual(result['error_type'], 'ValueError')
            self.assertIn('Unknown model', result['error'])


class BuildTests(unittest.TestCase):
    def test_notebooks_have_one_executable_cell_and_matching_sources(self):
        from tools.build_notebooks import build, SOURCE
        build()
        root = Path(__file__).resolve().parents[1]
        for name in ('Stem_Splitter_ComfyUI_Kaggle.ipynb', 'Stem_Splitter_Model_Packs.ipynb'):
            nb = json.loads((root / name).read_text())
            code = [c for c in nb['cells'] if c['cell_type'] == 'code']
            self.assertEqual(len(code), 1)
            tree = ast.parse(''.join(code[0]['source']))
            import base64,zlib
            packed = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('eN'))
            files = json.loads(zlib.decompress(base64.b64decode(packed)))
            for key, source in files.items():
                self.assertEqual(source, (root / key).read_text())
        self.assertEqual((root / 'Stem_Splitter_ComfyUI_Kaggle.ipynb').read_bytes(), (root / 'kaggle_roformer_stem_splitter.ipynb').read_bytes())

    def test_graphs_have_connected_separate_preview_and_export_stages(self):
        from tools.build_notebooks import workflow
        for kind in ('six', 'four', 'vocals'):
            graph = workflow(kind)
            self.assertEqual(len(graph['nodes']), 6)
            self.assertEqual(len(graph['links']), 5)
            for link in graph['links']:
                ident, a, slot, b, inp, typ = link
                self.assertIn(ident, graph['nodes'][a - 1]['outputs'][slot]['links'])
                self.assertEqual(graph['nodes'][b - 1]['inputs'][inp]['link'], ident)


if __name__ == '__main__':
    unittest.main()
