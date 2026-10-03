"""Trusted-gateway mode never runs Claude Code for Hermes auxiliary work.

After a capacity failure of an explicitly routed auxiliary model, Hermes retries the task once on
the session's live provider (``_try_main_agent_model_fallback``), with no setting to turn it off.
On a Claude session that provider is this plugin, so the plugin refuses any call made inside
Hermes' auxiliary call scope (``_relay_auxiliary_call``) before Claude Code starts. The main
agent's turns are untouched, and outside trusted-gateway mode the behaviour stays upstream's.
"""
import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk

FLAG = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_TRUSTED_GATEWAY'
GOOD = {FLAG: '1', 'ANTHROPIC_BASE_URL': 'https://gateway.example.test/anthropic', 'ANTHROPIC_API_KEY': 'gw-test-key-not-a-secret'}
REQUEST = dict(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])
OVERRIDES = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_FOUNDRY_API_KEY',
             'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_CUSTOM_HEADERS', 'CLAUDE_CODE_USE_BEDROCK',
             'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', FLAG)

# Stands in for native: it records that it started, then answers one turn without an upstream request.
NATIVE = r'''
import json, sys
open(sys.argv[1], 'w').close()
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
print(json.dumps({'type': 'assistant', 'message': {'id': 'm', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'MAIN'}]}}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'subtype': 'success', 'usage': {'input_tokens': 0, 'output_tokens': 0}}), flush=True)
'''


@pytest.fixture
def aux():
    from agent import auxiliary_client
    return auxiliary_client


@pytest.fixture
def native(monkeypatch, tmp_path):
    for key in OVERRIDES + ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
    script, marker = tmp_path / 'native.py', tmp_path / 'started'
    script.write_text(NATIVE, encoding='utf-8')

    def client(env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return directsdk.Client(command=[sys.executable, str(script), str(marker)], timeout=30)
    client.started = marker.exists
    return client


def _in_aux_scope(aux, task, call):
    """Run ``call`` inside Hermes' real auxiliary call scope, as ``call_llm`` does."""
    return aux._relay_auxiliary_call(lambda task: call())(task)


def test_trusted_gateway_refuses_an_auxiliary_call_before_claude_code_starts(aux, native):
    client = native(GOOD)
    with pytest.raises(directsdk.AuxiliaryCallRefused, match="'title_generation'"):
        _in_aux_scope(aux, 'title_generation', lambda: client.create(**REQUEST))
    with pytest.raises(directsdk.AuxiliaryCallRefused):
        _in_aux_scope(aux, 'compression', lambda: client.create(**REQUEST, stream=True))
    assert not native.started()
    assert aux._RELAY_AUX_CALL_CONTEXT.get() is None  # the scope closed; later main turns are clear


def test_trusted_gateway_refuses_an_async_auxiliary_call(aux, native):
    client = native(GOOD)

    @aux._relay_auxiliary_call_async
    async def call(task):
        # Hermes' async seam awaits create; the plugin hops to a worker thread with the context copied.
        return await client.create(**REQUEST)
    with pytest.raises(directsdk.AuxiliaryCallRefused, match="'session_search'"):
        asyncio.run(call('session_search'))
    assert not native.started()


def test_trusted_gateway_main_agent_turn_still_runs(aux, native):
    client = native(GOOD)
    assert aux._RELAY_AUX_CALL_CONTEXT.get() is None
    response = client.create(**REQUEST)
    assert response.choices[0].message.content == 'MAIN'
    assert native.started()


def test_off_auxiliary_call_keeps_upstream_behaviour(aux, native):
    client = native({})
    response = _in_aux_scope(aux, 'title_generation', lambda: client.create(**REQUEST))
    assert response.choices[0].message.content == 'MAIN'
    assert native.started()


def test_refusal_is_a_failed_fallback_candidate_not_a_retry(aux, native):
    """Hermes' auxiliary path records the refusal as a route-incompatible candidate: quarantined
    and skipped, never an auth refresh, a same-provider retry or a parameter-stripping rung."""
    from agent.auxiliary_fallback_recovery import send_with_parameter_rungs
    from agent.auxiliary_health import fallback_candidate_unavailable_reason

    client = native(GOOD)
    sends = []

    def send(c, kwargs):
        sends.append(kwargs)
        return c.create(**kwargs)
    with pytest.raises(directsdk.AuxiliaryCallRefused) as refused:
        _in_aux_scope(aux, 'title_generation', lambda: send_with_parameter_rungs(
            send, client, dict(REQUEST, temperature=0.3, max_tokens=64), task='title_generation'))
    assert len(sends) == 1
    error = refused.value
    assert fallback_candidate_unavailable_reason(error) == 'model incompatible with route'
    assert not aux._is_auth_error(error)
    assert not aux._is_rate_limit_error(error) and not aux._is_payment_error(error)
    assert not native.started()


def test_refused_main_agent_fallback_is_quarantined_and_skipped(aux, native, monkeypatch):
    """The F15 hop end to end: the main-agent fallback candidate is this plugin; Hermes' candidate
    helper gets the refusal, marks the lane unhealthy and returns None so the walk moves on."""
    client = native(GOOD)
    marked = []
    monkeypatch.setattr(aux, '_mark_provider_unhealthy', lambda provider, **kw: marked.append((provider, kw.get('reason'))))
    result = _in_aux_scope(aux, 'title_generation', lambda: aux._call_fallback_candidate_sync(
        client, 'sonnet', 'main-agent(claude-subscription-directsdk-experimental)', task='title_generation',
        messages=REQUEST['messages'], temperature=None, max_tokens=64, tools=None, effective_timeout=30,
        effective_extra_body={}, reasoning_config=None))
    assert result is None
    assert [reason for _, reason in marked] == ['model incompatible with route']
    assert not native.started()
