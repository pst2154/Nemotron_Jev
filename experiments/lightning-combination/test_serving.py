import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import serve_systemone


class ServingTests(unittest.TestCase):
    def test_adapter_settings(self):
        with patch.dict(os.environ, {'SERVER_PORT': '8310', 'SYSTEMONE_PORT': '8791'}, clear=True):
            env = serve_systemone.adapter_environment()
        self.assertEqual(env['UPSTREAM_BASE_URL'], 'http://127.0.0.1:8310/v1')
        self.assertEqual(env['UPSTREAM_MODEL'], 'trained')
        self.assertEqual(env['PORT'], '8791')
        self.assertEqual(env['HOST'], '127.0.0.1')
        for key in ['STATE_FIRST', 'PERSISTENT_UPSTREAM', 'ISOLATE_REQUEST_CACHE']:
            self.assertEqual(env[key], '1')

    def test_readiness(self):
        process = Mock(); process.poll.return_value = None
        response = Mock(); response.status = 200
        with patch('serve_systemone.urllib.request.urlopen') as opener:
            opener.return_value.__enter__.return_value = response
            serve_systemone.await_model(process, 'http://127.0.0.1/health', timeout=1)
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            serve_systemone.await_model(process, 'http://127.0.0.1/health', timeout=1)

    def test_supervisor_stops_api_when_model_exits(self):
        model = Mock(); model.poll.side_effect = [None, 1, 1]
        api = Mock(); api.poll.return_value = None
        with patch('serve_systemone.subprocess.Popen', side_effect=[model, api]), \
             patch('serve_systemone.await_model'), patch('serve_systemone.time.sleep'), \
             patch('serve_systemone.signal.signal'):
            with self.assertRaisesRegex(RuntimeError, 'serving process exited'):
                serve_systemone.main()
        api.terminate.assert_called_once()
        model.wait.assert_called_once()
        api.wait.assert_called_once()

    def launch(self, extra):
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, 'MODEL_DIR': temp, 'ADAPTER_DIR': temp,
                   'RUN_DIR': temp, 'GPU_UUID': 'test-device', **extra}
            env.pop('SYSTEMONE_API_KEY', None)
            cmd = 'docker() { printf "%s\\n" "$@"; }; export -f docker; bash "$1"'
            return subprocess.run(['bash', '-c', cmd, 'test', str(Path(__file__).with_name('launch_combination.sh'))],
                                  env=env, capture_output=True, text=True)

    def test_combined_launch_arguments(self):
        result = self.launch({'SYSTEMONE_ENABLED': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn('--entrypoint', args)
        self.assertIn('/app/serve_systemone.py', args)
        self.assertIn('trained=/adapter', args)
        self.assertIn('SYSTEMONE_PORT=8790', args)
        self.assertIn('SYSTEMONE_API_KEY', args)
        self.assertIn('16', args)

    def test_original_launch_remains_model_only(self):
        result = self.launch({'SYSTEMONE_ENABLED': '0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--entrypoint', result.stdout.splitlines())

    def test_rejects_unauthenticated_external_binding(self):
        result = self.launch({'SYSTEMONE_ENABLED': '1', 'SYSTEMONE_HOST': '0.0.0.0'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SYSTEMONE_API_KEY is required', result.stderr)

    def test_rejects_conflicting_ports(self):
        result = self.launch({'SYSTEMONE_ENABLED': '1', 'SERVER_PORT': '8790'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ports must differ', result.stderr)


if __name__ == '__main__':
    unittest.main()
