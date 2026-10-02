"""The admission relay reaches its upstream through the environment's egress proxy.

A sandbox may resolve no public names and reach the internet only through an HTTP proxy
(``HTTPS_PROXY`` with ``NO_PROXY``), which native honours. The relay tunnels with CONNECT the same
way, so the upstream's name is resolved by the proxy, never by the relay. No test reaches a real
host: the upstream is ``gateway.invalid`` (never resolvable), served by a loopback TLS stub that a
loopback CONNECT proxy fixture stands in front of.
"""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import threading
import urllib.request
from urllib.parse import urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import admission

PROXY_VARS = ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy')


@pytest.fixture
def no_proxy_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --- the environment's proxy -----------------------------------------------------------------

def test_no_proxy_settings_mean_a_direct_connection(no_proxy_env):
    assert admission.proxy_tunnel(urlsplit('https://gateway.example.test/anthropic')) is None


def test_no_proxy_bypasses_the_tunnel(no_proxy_env):
    no_proxy_env.setenv('HTTPS_PROXY', 'http://proxy.example.test:9445')
    no_proxy_env.setenv('NO_PROXY', 'localhost,127.0.0.1,gateway.example.test')
    assert admission.proxy_tunnel(urlsplit('https://gateway.example.test/anthropic')) is None


def test_the_tunnel_carries_the_proxys_decoded_credentials(no_proxy_env):
    no_proxy_env.setenv('https_proxy', 'http://us%40er:p%3Ass@proxy.example.test:9445')
    no_proxy_env.setenv('no_proxy', 'localhost,127.0.0.1')
    host, port, headers = admission.proxy_tunnel(urlsplit('https://gateway.example.test'))
    assert (host, port) == ('proxy.example.test', 9445)
    assert headers == {'Proxy-Authorization': 'Basic ' + base64.b64encode(b'us@er:p:ss').decode()}


def test_all_proxy_counts_and_a_proxy_without_credentials_sends_none(no_proxy_env):
    no_proxy_env.setenv('ALL_PROXY', 'proxy.example.test:3128')
    assert admission.proxy_tunnel(urlsplit('https://gateway.example.test')) == ('proxy.example.test', 3128, {})


@pytest.mark.parametrize('proxy', ['https://secret-user:secret-pass@proxy.example.test:9445',
                                   'socks5://secret-user:secret-pass@proxy.example.test:1080',
                                   'http://secret-user:secret-pass@proxy.example.test:notaport'])
def test_a_proxy_the_relay_cannot_tunnel_through_is_refused_without_echoing_it(no_proxy_env, proxy):
    no_proxy_env.setenv('HTTPS_PROXY', proxy)
    with pytest.raises(ValueError) as caught:
        admission.proxy_tunnel(urlsplit('https://gateway.example.test'))
    assert 'HTTPS_PROXY' in str(caught.value)
    assert 'secret' not in str(caught.value)


# --- the relay through a CONNECT proxy -------------------------------------------------------

def _certificate(tmp_path, name):
    openssl = shutil.which('openssl')
    if not openssl:
        pytest.skip('openssl is needed to mint the loopback TLS fixture certificate')
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    run = subprocess.run([openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                          '-subj', '/CN=' + name, '-addext', 'subjectAltName=DNS:' + name,
                          '-keyout', str(key), '-out', str(cert)], capture_output=True, text=True)
    if run.returncode != 0:
        pytest.skip('this openssl cannot mint the fixture certificate: ' + run.stderr.strip()[:200])
    return cert, key


@pytest.fixture
def plain_ssl():
    """Hermes core may have injected truststore into ``ssl``; the fixture needs stdlib contexts."""
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


class _ConnectProxy(socketserver.ThreadingTCPServer):
    """A loopback CONNECT proxy that requires Basic credentials and maps every tunnel to the stub."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, upstream_port, credentials):
        self.upstream_port = upstream_port
        self.expected = 'Basic ' + base64.b64encode(credentials.encode()).decode()
        self.connects = []
        super().__init__(('127.0.0.1', 0), _ConnectHandler)


class _ConnectHandler(socketserver.StreamRequestHandler):
    def handle(self):
        request = self.rfile.readline().decode('latin-1').strip()
        headers = {}
        while True:
            line = self.rfile.readline().decode('latin-1').strip()
            if not line:
                break
            name, _, value = line.partition(':')
            headers[name.strip().lower()] = value.strip()
        method, _, rest = request.partition(' ')
        target = rest.split(' ')[0]
        self.server.connects.append({'method': method, 'target': target, 'auth': headers.get('proxy-authorization')})
        if method != 'CONNECT':
            self.wfile.write(b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n')
            return
        if headers.get('proxy-authorization') != self.server.expected:
            self.wfile.write(b'HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n')
            return
        upstream = socket.create_connection(('127.0.0.1', self.server.upstream_port))
        self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
        self.wfile.flush()

        def pipe(src, dst):
            try:
                while True:
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    dst.sendall(chunk)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        back = threading.Thread(target=pipe, args=(upstream, self.connection), daemon=True)
        back.start()
        pipe(self.connection, upstream)
        back.join(timeout=5)
        upstream.close()


def test_the_relay_tunnels_through_the_proxy_and_never_resolves_the_upstream(no_proxy_env, plain_ssl, tmp_path):
    cert, key = _certificate(tmp_path, 'gateway.invalid')
    seen = []
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append({'path': self.path, 'host': self.headers.get('Host'), 'key': self.headers.get('x-api-key'),
                         'proxy-authorization': self.headers.get('Proxy-Authorization'),
                         'body': self.rfile.read(int(self.headers['Content-Length']))})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type': 'message_start', 'message': {'id': 'm', 'role': 'assistant', 'model': 'claude-sonnet-5-5', 'content': [], 'usage': usage}},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': usage},
                {'type': 'message_stop'},
            ]
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    gateway = ThreadingHTTPServer(('127.0.0.1', 0), Gateway)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(str(cert), str(key))
    gateway.socket = tls.wrap_socket(gateway.socket, server_side=True)
    proxy = _ConnectProxy(gateway.server_port, 'box-user:p@ss')
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (gateway, proxy)]
    for t in threads:
        t.start()
    # The relay verifies TLS as always: the upstream's own name, against the fixture's trust root.
    default_context = ssl.create_default_context
    no_proxy_env.setattr(admission.ssl, 'create_default_context', lambda *a, **k: default_context(cafile=str(cert)))
    no_proxy_env.setenv('HTTPS_PROXY', f'http://box-user:p%40ss@127.0.0.1:{proxy.server_address[1]}')
    no_proxy_env.setenv('NO_PROXY', 'localhost,127.0.0.1')

    gate = admission.Admission('https://gateway.invalid/anthropic', timeout=10)
    body = b'{"model":"claude-sonnet-5-5","max_tokens":8,"messages":[{"role":"user","content":"fixture"}]}'
    try:
        request = urllib.request.Request(gate.url + '/v1/messages', data=body, method='POST', headers={
            'Content-Type': 'application/json', 'x-api-key': 'gw-test-key-not-a-secret', 'anthropic-version': '2023-06-01'})
        # Native reaches the loopback relay directly (NO_PROXY), as in a sandbox.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=10) as response:
            status, answer = response.status, response.read()
    finally:
        gate.close()
        for server in (gateway, proxy):
            server.shutdown()
            server.server_close()

    assert gate.failure is None and status == 200 and gate.status == 200
    assert b'message_stop' in answer
    assert proxy.connects == [{'method': 'CONNECT', 'target': 'gateway.invalid:443',
                               'auth': 'Basic ' + base64.b64encode(b'box-user:p@ss').decode()}]
    assert len(seen) == 1
    assert seen[0]['path'] == '/anthropic/v1/messages'
    assert seen[0]['host'] == 'gateway.invalid'
    assert seen[0]['key'] == 'gw-test-key-not-a-secret'
    assert seen[0]['proxy-authorization'] is None  # proxy credentials stay on the CONNECT, never inside the tunnel
    assert seen[0]['body'] == body


def test_a_refusing_proxy_fails_the_call_without_reaching_the_upstream(no_proxy_env, tmp_path):
    proxy = _ConnectProxy(9, 'right:credentials')
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    no_proxy_env.setenv('HTTPS_PROXY', f'http://wrong:credentials@127.0.0.1:{proxy.server_address[1]}')
    no_proxy_env.setenv('NO_PROXY', 'localhost,127.0.0.1')
    gate = admission.Admission('https://gateway.invalid/anthropic', timeout=5)
    try:
        request = urllib.request.Request(gate.url + '/v1/messages', data=b'{}', method='POST',
                                         headers={'Content-Type': 'application/json'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            opener.open(request, timeout=10).read()
        except Exception:
            pass  # The relay closes the connection: what matters is the recorded failure below.
    finally:
        gate.close()
        proxy.shutdown()
        proxy.server_close()
    assert gate.failure == 'OSError'  # http.client: "Tunnel connection failed: 407 ..."
    assert gate.status is None
    assert [c['method'] for c in proxy.connects] == ['CONNECT']
