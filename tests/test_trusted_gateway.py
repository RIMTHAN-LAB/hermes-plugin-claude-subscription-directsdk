"""Opt-in trusted-gateway mode: native in API-key mode against an operator-run HTTPS gateway.

Off (the default) the inherited-environment guard is unchanged. On, it admits exactly
ANTHROPIC_BASE_URL (https) and ANTHROPIC_API_KEY, and the relay's single upstream request goes to
that gateway with native's own headers, key and body. No test reaches a real host: the gateway is
a loopback TLS stub with a throwaway self-signed certificate.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import admission
import directsdk

FLAG = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_TRUSTED_GATEWAY'
KEY = 'gw-test-key-not-a-secret'
OVERRIDES = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_FOUNDRY_API_KEY',
             'CLAUDE_CODE_OAUTH_TOKEN', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', FLAG)
GOOD = {FLAG: '1', 'ANTHROPIC_BASE_URL': 'https://gateway.example.test/anthropic', 'ANTHROPIC_API_KEY': KEY}
REQUEST = dict(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])

# Stands in for the unmodified CLI in API-key mode: it sends its own identity headers and the key
# it inherited to whatever ANTHROPIC_BASE_URL it was given, and tries a second (recovery) request.
NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
base = os.environ['ANTHROPIC_BASE_URL']
headers = {'Content-Type': 'application/json', 'x-api-key': os.environ.get('ANTHROPIC_API_KEY', ''),
           'anthropic-version': '2023-06-01', 'anthropic-beta': 'fixture-beta-2026-01-01',
           'User-Agent': 'claude-cli/9.9.9 (external, sdk-cli)', 'x-app': 'cli',
           'X-Fixture-Child-Base': base}
for _ in range(2):
    try:
        urllib.request.urlopen(urllib.request.Request(base + '/v1/messages', data=os.environ['FIXTURE_BODY'].encode(), headers=headers), timeout=5).read()
    except urllib.error.HTTPError:
        break
print(json.dumps({'type': 'assistant', 'message': {'id': 'first', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'FIRST'}]}}))
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'usage': {'input_tokens': 0, 'output_tokens': 0}}))
'''


@pytest.fixture
def clean_env(monkeypatch):
    for key in OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --- the guard -------------------------------------------------------------------------------

def test_off_is_none_whatever_else_is_set():
    assert admission.trusted_gateway({}) is None
    assert admission.trusted_gateway({**GOOD, FLAG: '0'}) is None
    assert admission.trusted_gateway({**GOOD, FLAG: 'true'}) is None  # only "1" turns it on
    assert admission.trusted_gateway({**GOOD, FLAG: ''}) is None


def test_on_admits_exactly_base_url_and_key():
    assert admission.trusted_gateway(GOOD) == 'https://gateway.example.test/anthropic'
    assert admission.trusted_gateway({**GOOD, 'ANTHROPIC_BASE_URL': 'https://127.0.0.1:8443'}) == 'https://127.0.0.1:8443'
    assert admission.trusted_gateway({**GOOD, 'CLAUDE_CODE_USE_BEDROCK': '0'}) == GOOD['ANTHROPIC_BASE_URL']


@pytest.mark.parametrize('url', [
    'http://gateway.example.test', 'http://127.0.0.1:8080', 'https://user:secret-pass@gateway.example.test',
    'https://secret-user@gateway.example.test', 'https://gateway.example.test/?key=secret-q',
    'https://gateway.example.test/?', 'https://gateway.example.test/#secret-frag', 'https://gateway.example.test#',
    'https://', 'https://gateway.example.test:notaport', 'gateway.example.test', 'ftp://gateway.example.test',
])
def test_on_refuses_a_bad_base_url_without_echoing_it(url):
    with pytest.raises(ValueError) as caught:
        admission.trusted_gateway({**GOOD, 'ANTHROPIC_BASE_URL': url})
    message = str(caught.value)
    assert 'ANTHROPIC_BASE_URL' in message and 'https' in message
    assert url not in message and 'secret' not in message and KEY not in message


@pytest.mark.parametrize('missing', ['ANTHROPIC_BASE_URL', 'ANTHROPIC_API_KEY'])
@pytest.mark.parametrize('blank', [None, '', '   '])
def test_on_refuses_a_missing_url_or_key(missing, blank):
    env = dict(GOOD)
    if blank is None:
        del env[missing]
    else:
        env[missing] = blank
    with pytest.raises(ValueError, match=f'requires {missing}'):
        admission.trusted_gateway(env)


@pytest.mark.parametrize('key,value', [
    ('ANTHROPIC_AUTH_TOKEN', 'secret-bearer'), ('ANTHROPIC_FOUNDRY_API_KEY', 'secret-foundry'),
    ('CLAUDE_CODE_OAUTH_TOKEN', 'secret-oauth'),
    ('CLAUDE_CODE_USE_BEDROCK', '1'), ('CLAUDE_CODE_USE_VERTEX', 'true'), ('CLAUDE_CODE_USE_FOUNDRY', 'yes'),
])
def test_on_still_refuses_other_credentials_and_backends(key, value):
    with pytest.raises(ValueError) as caught:
        admission.trusted_gateway({**GOOD, key: value})
    message = str(caught.value)
    assert key in message and KEY not in message and 'secret' not in message


# --- the transport ---------------------------------------------------------------------------

@pytest.mark.parametrize('key', ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'])
def test_off_inherited_overrides_are_refused_as_before(clean_env, key):
    clean_env.setenv(key, 'https://gateway.example.test' if key == 'ANTHROPIC_BASE_URL' else 'secret-value')
    with pytest.raises(ValueError, match='OAuth provider refuses conflicting native auth/backend overrides: ' + key):
        directsdk.Client(command='/does/not/exist').create(**REQUEST)


def test_off_flag_not_exactly_one_keeps_the_subscription_guard(clean_env):
    for key, value in {**GOOD, FLAG: 'true'}.items():
        clean_env.setenv(key, value)
    with pytest.raises(ValueError, match='OAuth provider refuses'):
        directsdk.Client(command='/does/not/exist').create(**REQUEST)


@pytest.mark.parametrize('change', [
    {'ANTHROPIC_BASE_URL': 'http://gateway.example.test'}, {'ANTHROPIC_API_KEY': ''},
    {'ANTHROPIC_AUTH_TOKEN': 'secret-bearer'}, {'CLAUDE_CODE_USE_VERTEX': '1'},
])
def test_on_misconfigured_is_refused_before_native_runs(clean_env, tmp_path, change):
    marker = tmp_path / 'ran'
    native = tmp_path / 'native.py'
    native.write_text(f'open({str(marker)!r}, "w").close()\n')
    for key, value in {**GOOD, **change}.items():
        clean_env.setenv(key, value)
    with pytest.raises(ValueError, match=FLAG) as caught:
        directsdk.Client(command=[sys.executable, str(native)]).create(**REQUEST)
    assert 'secret' not in str(caught.value) and KEY not in str(caught.value)
    assert not marker.exists()


def test_on_explicit_env_is_held_to_the_same_rule(tmp_path):
    env = {'PATH': os.defpath, 'HOME': str(tmp_path), **GOOD, 'ANTHROPIC_BASE_URL': 'http://127.0.0.1:9'}
    with pytest.raises(ValueError, match=FLAG):
        directsdk.Client(command='/does/not/exist', env=env).create(**REQUEST)


def _certificate(tmp_path):
    openssl = shutil.which('openssl')
    if not openssl:
        pytest.skip('openssl is needed to mint the loopback TLS fixture certificate')
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    run = subprocess.run([openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                          '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                          '-keyout', str(key), '-out', str(cert)], capture_output=True, text=True)
    if run.returncode != 0:
        pytest.skip('this openssl cannot mint the fixture certificate: ' + run.stderr.strip()[:200])
    return cert, key


@pytest.fixture
def plain_ssl():
    """Hermes core injects truststore into ``ssl`` (OS trust store, no cafile); the fixture needs
    stdlib contexts for its own trust root, so lift the injection for this test only."""
    try:
        import truststore
    except ImportError:
        yield
        return
    injected = ssl.SSLContext is truststore.SSLContext
    if injected:
        truststore.extract_from_ssl()
    try:
        yield
    finally:
        if injected:
            truststore.inject_into_ssl()


def test_on_relay_forwards_once_to_the_gateway_with_native_headers_key_and_body(clean_env, plain_ssl, tmp_path):
    cert, key = _certificate(tmp_path)
    seen = []
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append({'path': self.path, 'headers': {k.lower(): v for k, v in self.headers.items()},
                         'body': self.rfile.read(int(self.headers['Content-Length']))})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type': 'message_start', 'message': {'id': 'first', 'role': 'assistant', 'model': 'claude-sonnet-5-5', 'content': [], 'usage': usage}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'FIRST'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': usage},
                {'type': 'message_stop'},
            ]
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Gateway)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(str(cert), str(key))
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The relay verifies TLS as always; only the trust root is the fixture's own certificate.
    default_context = ssl.create_default_context
    clean_env.setattr(admission.ssl, 'create_default_context', lambda *a, **k: default_context(cafile=str(cert)))

    body = '{"model":"claude-sonnet-5-5","max_tokens":64,"messages":[{"role":"user","content":"fixture"}]}'
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    gateway = f'https://127.0.0.1:{server.server_port}/anthropic'
    for name, value in {FLAG: '1', 'ANTHROPIC_BASE_URL': gateway, 'ANTHROPIC_API_KEY': KEY, 'FIXTURE_BODY': body}.items():
        clean_env.setenv(name, value)
    client = directsdk.Client(command=[sys.executable, str(native)])
    try:
        result = client.create(**REQUEST)
    finally:
        client.close()
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.choices[0].message.content == 'FIRST'
    assert len(seen) == 1, 'exactly one upstream request per Hermes call; native recovery is refused locally'
    request = seen[0]
    assert request['path'] == '/anthropic/v1/messages'
    headers = request['headers']
    assert headers['x-api-key'] == KEY  # the per-run gateway key, passed through untouched
    assert 'authorization' not in headers
    assert headers['user-agent'] == 'claude-cli/9.9.9 (external, sdk-cli)'
    assert headers['x-app'] == 'cli'
    assert headers['anthropic-version'] == '2023-06-01'
    assert headers['anthropic-beta'] == 'fixture-beta-2026-01-01'
    # Native itself was pointed at the loopback relay, never straight at the gateway.
    assert headers['x-fixture-child-base'].startswith('http://127.0.0.1:')
    assert '/admit/' in headers['x-fixture-child-base']
    assert request['body'] == body.encode()
    assert result.usage.model_dump()['native_admission']['upstream_requests'] == 1
    assert result.usage.model_dump()['native_admission']['blocked_requests'] == 1


# --- discovery -------------------------------------------------------------------------------

def test_discovery_in_gateway_mode(profile, tmp_path):
    """The picker handshake sends no Messages request; in gateway mode its relay points at the
    gateway, and a misconfigured gateway degrades to the pinned catalog instead of failing setup."""
    from test_directsdk_setup import PINNED_PICKER, PRO, _cli
    command, env = _cli(tmp_path, {**PRO, 'models': PINNED_PICKER})
    env = {k: v for k, v in env.items() if k not in OVERRIDES}
    rows = profile.discover_models(command=command, env={**env, **GOOD})
    assert rows and all(row['upstream_requests'] == 0 for row in rows)
    bad = {**env, **GOOD, 'ANTHROPIC_BASE_URL': 'http://gateway.example.test'}
    assert profile.discover_models(command=command, env=bad) is None
