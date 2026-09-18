import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from anytrain.chat import Chat


def _chat(client: object) -> Chat:
    with (
        patch("anytrain.chat.chat.config_from_env", return_value=SimpleNamespace(model="test", models=())),
        patch("anytrain.chat.chat.create_client", return_value=client),
    ):
        return Chat()


def test_sync_context_closes_once_and_rejects_reuse() -> None:
    provider = Mock()
    chat = _chat(provider)
    with pytest.raises(ValueError, match="body"), chat:
        raise ValueError("body")
    chat.close()
    provider.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        chat.request("hello")


def test_async_context_closes_both_clients_once() -> None:
    sync = Mock()
    asynchronous = SimpleNamespace(close=AsyncMock())
    chat = _chat(sync)

    async def run() -> None:
        async with chat:
            chat._async_client = asynchronous
            chat._async_client_loop = asyncio.get_running_loop()
            with pytest.raises(RuntimeError, match="aclose"):
                chat.close()
        await chat.aclose()
        chat.close()
        asynchronous.close.assert_awaited_once_with()

    asyncio.run(run())
    sync.close.assert_called_once_with()


def test_shared_glm_client_is_closed_once() -> None:
    provider = Mock()
    chat = _chat(provider)
    chat._async_client = provider
    asyncio.run(chat.aclose())
    chat.close()
    provider.close.assert_called_once_with()


def test_async_close_failure_still_closes_sync_client_and_can_retry() -> None:
    sync = Mock()
    asynchronous = SimpleNamespace(close=AsyncMock(side_effect=[ValueError("close failed"), None]))
    chat = _chat(sync)
    chat._async_client = asynchronous

    async def run() -> None:
        with pytest.raises(ValueError, match="close failed"):
            await chat.aclose()
        sync.close.assert_called_once_with()
        with pytest.raises(RuntimeError, match="closed"):
            await chat.async_request("hello")
        await chat.aclose()
        assert asynchronous.close.await_count == 2

    asyncio.run(run())
    sync.close.assert_called_once_with()


def test_async_close_keeps_lock_until_cleanup_finishes_after_cancellation() -> None:
    sync = Mock()
    chat = _chat(sync)

    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def close() -> None:
            started.set()
            await release.wait()

        asynchronous = SimpleNamespace(close=AsyncMock(side_effect=close))
        chat._async_client = asynchronous
        task = asyncio.create_task(chat.aclose())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert chat._lock.locked()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        assert not chat._lock.locked()
        await chat.aclose()
        asynchronous.close.assert_awaited_once_with()

    asyncio.run(run())
    sync.close.assert_called_once_with()


def test_sync_close_in_event_loop_rejects_contended_lock_without_blocking() -> None:
    chat = _chat(Mock())

    async def run() -> None:
        chat._lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="in-flight"):
                chat.close()
        finally:
            chat._lock.release()
        await chat.aclose()

    asyncio.run(run())


def test_cancelled_glm_request_finishes_before_client_closes() -> None:
    from threading import Event

    from anytrain.chat import ModelType

    events: list[str] = []
    started, release = Event(), Event()
    provider = Mock()
    provider.close.side_effect = lambda: events.append("close")
    chat = _chat(provider)
    chat.model_type = ModelType.GLM

    def request(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        events.append("request_done")
        return Mock(text="done")

    async def run() -> None:
        pending = asyncio.create_task(chat.async_request("hello"))
        closing: asyncio.Task[None] | None = None
        try:
            assert await asyncio.to_thread(started.wait, 2)
            pending.cancel()
            await asyncio.sleep(0)
            closing = asyncio.create_task(chat.aclose())
            await asyncio.sleep(0)
            assert chat._lock.locked()
            assert not events
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
            if closing is not None:
                await asyncio.wait_for(closing, timeout=2)
        assert events == ["request_done", "close"]
        assert chat.last_response is None

    with patch("anytrain.chat._request.request_detail", side_effect=request):
        asyncio.run(run())
