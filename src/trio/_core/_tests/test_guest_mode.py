from __future__ import annotations

import asyncio
import contextlib
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from functools import partial
from math import inf
from typing import (
    TYPE_CHECKING,
    NoReturn,
    TypeAlias,
    TypeVar,
    cast,
)

import pytest
import sniffio
from outcome import Outcome

import trio
import trio.testing
from trio.abc import Clock, Instrument

from .tutil import gc_collect_harder, restore_unraisablehook

if TYPE_CHECKING:
    from trio._channel import MemorySendChannel

T = TypeVar("T")
InHost: TypeAlias = Callable[[Callable[[], object]], None]


# The simplest possible "host" loop.
# Nice features:
# - we can run code "outside" of trio using the schedule function passed to
#   our main
# - final result is returned
# - any unhandled exceptions cause an immediate crash
def trivial_guest_run(
    trio_fn: Callable[[InHost], Awaitable[T]],
    *,
    in_host_after_start: Callable[[], None] | None = None,
    host_uses_signal_set_wakeup_fd: bool = False,
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> T:
    todo: queue.Queue[tuple[str, Outcome[T] | Callable[[], object]]] = queue.Queue()

    host_thread = threading.current_thread()

    def run_sync_soon_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is threading.current_thread():  # pragma: no cover
            crash = partial(
                pytest.fail,
                "run_sync_soon_threadsafe called from host thread",
            )
            todo.put(("run", crash))
        todo.put(("run", fn))

    def run_sync_soon_not_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is not threading.current_thread():  # pragma: no cover
            crash = partial(
                pytest.fail,
                "run_sync_soon_not_threadsafe called from worker thread",
            )
            todo.put(("run", crash))
        todo.put(("run", fn))

    def done_callback(outcome: Outcome[T]) -> None:
        nonlocal todo
        todo.put(("unwrap", outcome))

    trio.lowlevel.start_guest_run(
        trio_fn,
        run_sync_soon_not_threadsafe,
        run_sync_soon_threadsafe=run_sync_soon_threadsafe,
        run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe,
        done_callback=done_callback,
        host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd,
        clock=clock,
        instruments=instruments,
        restrict_keyboard_interrupt_to_checkpoints=restrict_keyboard_interrupt_to_checkpoints,
        strict_exception_groups=strict_exception_groups,
    )
    if in_host_after_start is not None:
        in_host_after_start()

    try:
        while True:
            op, obj = todo.get()
            if op == "run":
                assert not isinstance(obj, Outcome)
                obj()
            elif op == "unwrap":
                assert isinstance(obj, Outcome)
                return obj.unwrap()
            else:  # pragma: no cover
                raise NotImplementedError(f"{op!r} not handled")
    finally:
        # Make sure that exceptions raised here don't capture these, so that
        # if an exception does cause us to abandon a run then the Trio state
        # has a chance to be GC'ed and warn about it.
        del todo, run_sync_soon_threadsafe, done_callback


def test_guest_trivial() -> None:
    async def trio_return(in_host: InHost) -> str:
        await trio.lowlevel.checkpoint()
        return "ok"

    assert trivial_guest_run(trio_return) == "ok"

    async def trio_fail(in_host: InHost) -> NoReturn:
        raise KeyError("whoopsiedaisy")

    with pytest.raises(KeyError, match="whoopsiedaisy"):
        trivial_guest_run(trio_fail)


def test_guest_can_do_io() -> None:
    async def trio_main(in_host: InHost) -> None:
        record = []
        a, b = trio.socket.socketpair()
        with a, b:
            async with trio.open_nursery() as nursery:

                async def do_receive() -> None:
                    record.append(await a.recv(1))

                nursery.start_soon(do_receive)
                await trio.testing.wait_all_tasks_blocked()

                await b.send(b"x")

        assert record == [b"x"]

    trivial_guest_run(trio_main)


def test_guest_is_initialized_when_start_returns() -> None:
    trio_token = None
    record = []

    async def trio_main(in_host: InHost) -> str:
        record.append("main task ran")
        await trio.lowlevel.checkpoint()
        assert trio.lowlevel.current_trio_token() is trio_token
        return "ok"

    def after_start() -> None:
        # We should get control back before the main task executes any code
        assert record == []

        nonlocal trio_token
        trio_token = trio.lowlevel.current_trio_token()
        trio_token.run_sync_soon(record.append, "run_sync_soon cb ran")

        @trio.lowlevel.spawn_system_task
        async def early_task() -> None:
            record.append("system task ran")
            await trio.lowlevel.checkpoint()

    res = trivial_guest_run(trio_main, in_host_after_start=after_start)
    assert res == "ok"
    assert set(record) == {"system task ran", "main task ran", "run_sync_soon cb ran"}

    class BadClock(Clock):
        def start_clock(self) -> NoReturn:
            raise ValueError("whoops")

        def current_time(self) -> float:
            raise NotImplementedError()

        def deadline_to_sleep_time(self, deadline: float) -> float:
            raise NotImplementedError()

    def after_start_never_runs() -> None:  # pragma: no cover
        pytest.fail("shouldn't get here")

    # Errors during initialization (which can only be TrioInternalErrors)
    # are raised out of start_guest_run, not out of the done_callback
    with pytest.raises(trio.TrioInternalError):
        trivial_guest_run(
            trio_main,
            clock=BadClock(),
            in_host_after_start=after_start_never_runs,
        )


def test_host_can_directly_wake_trio_task() -> None:
    async def trio_main(in_host: InHost) -> str:
        ev = trio.Event()
        in_host(ev.set)
        await ev.wait()
        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


def test_host_altering_deadlines_wakes_trio_up() -> None:
    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        with trio.CancelScope() as cscope:
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep_forever()
        assert cscope.cancelled_caught

        with trio.CancelScope() as cscope:
            # also do a change that doesn't affect the next deadline, just to
            # exercise that path
            in_host(lambda: set_deadline(cscope, 1e6))
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep(999)
        assert cscope.cancelled_caught

        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


def test_guest_mode_sniffio_integration() -> None:
    current_async_library = sniffio.current_async_library
    sniffio_library = sniffio.thread_local

    async def trio_main(in_host: InHost) -> str:
        async def synchronize() -> None:
            """Wait for all in_host() calls issued so far to complete."""
            evt = trio.Event()
            in_host(evt.set)
            await evt.wait()

        # Host and guest have separate sniffio_library contexts
        in_host(partial(setattr, sniffio_library, "name", "nullio"))
        await synchronize()
        assert current_async_library() == "trio"

        record = []
        in_host(lambda: record.append(current_async_library()))
        await synchronize()
        assert record == ["nullio"]
        assert current_async_library() == "trio"

        return "ok"

    try:
        assert trivial_guest_run(trio_main) == "ok"
    finally:
        sniffio_library.name = None


def test_guest_mode_trio_context_detection() -> None:
    def check(thing: bool) -> None:
        assert thing

    assert not trio.lowlevel.in_trio_run()
    assert not trio.lowlevel.in_trio_task()

    async def trio_main(in_host: InHost) -> None:
        for _ in range(2):
            assert trio.lowlevel.in_trio_run()
            assert trio.lowlevel.in_trio_task()

            in_host(lambda: check(trio.lowlevel.in_trio_run()))
            in_host(lambda: check(not trio.lowlevel.in_trio_task()))

    trivial_guest_run(trio_main)
    assert not trio.lowlevel.in_trio_run()
    assert not trio.lowlevel.in_trio_task()


def test_warn_set_wakeup_fd_overwrite() -> None:
    assert signal.set_wakeup_fd(-1) == -1

    async def trio_main(in_host: InHost) -> str:
        return "ok"

    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)

        # Warn if there's already a wakeup fd
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match="signal handling code.*collided"):
                assert trivial_guest_run(trio_main) == "ok"
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()

        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match="signal handling code.*collided"):
                assert (
                    trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=False)
                    == "ok"
                )
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()

        # Don't warn if there isn't already a wakeup fd
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert trivial_guest_run(trio_main) == "ok"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=True)
                == "ok"
            )

        # If there's already a wakeup fd, but we've been told to trust it,
        # then it's left alone and there's no warning
        signal.set_wakeup_fd(a.fileno())
        try:

            async def trio_check_wakeup_fd_unaltered(in_host: InHost) -> str:
                fd = signal.set_wakeup_fd(-1)
                assert fd == a.fileno()
                signal.set_wakeup_fd(fd)
                return "ok"

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    trivial_guest_run(
                        trio_check_wakeup_fd_unaltered,
                        host_uses_signal_set_wakeup_fd=True,
                    )
                    == "ok"
                )
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()


def test_host_wakeup_doesnt_trigger_wait_all_tasks_blocked() -> None:
    # This is designed to hit the branch in unrolled_run where:
    #   idle_primed=True
    #   runner.runq is empty
    #   events is Truth-y
    # ...and confirm that in this case, wait_all_tasks_blocked does not get
    # triggered.
    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        print(f"setting deadline {new_deadline}")
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        async def sit_in_wait_all_tasks_blocked(watb_cscope: trio.CancelScope) -> None:
            with watb_cscope:
                # Overall point of this test is that this
                # wait_all_tasks_blocked should *not* return normally, but
                # only by cancellation.
                await trio.testing.wait_all_tasks_blocked(cushion=9999)
                raise AssertionError(  # pragma: no cover
                    "wait_all_tasks_blocked should *not* return normally, "
                    "only by cancellation.",
                )
            assert watb_cscope.cancelled_caught

        async def get_woken_by_host_deadline(watb_cscope: trio.CancelScope) -> None:
            with trio.CancelScope() as cscope:
                print("scheduling stuff to happen")

                # Altering the deadline from the host, to something in the
                # future, will cause the run loop to wake up, but then
                # discover that there is nothing to do and go back to sleep.
                # This should *not* trigger wait_all_tasks_blocked.
                #
                # So the 'before_io_wait' here will wait until we're blocking
                # with the wait_all_tasks_blocked primed, and then schedule a
                # deadline change. The critical test is that this should *not*
                # wake up 'sit_in_wait_all_tasks_blocked'.
                #
                # The after we've had a chance to wake up
                # 'sit_in_wait_all_tasks_blocked', we want the test to
                # actually end. So in after_io_wait we schedule a second host
                # call to tear things down.
                class InstrumentHelper(Instrument):
                    def __init__(self) -> None:
                        self.primed = False

                    def before_io_wait(self, timeout: float) -> None:
                        print(f"before_io_wait({timeout})")
                        if timeout == 9999:  # pragma: no branch
                            assert not self.primed
                            in_host(lambda: set_deadline(cscope, 1e9))
                            self.primed = True

                    def after_io_wait(self, timeout: float) -> None:
                        if self.primed:  # pragma: no branch
                            print("instrument triggered")
                            in_host(lambda: cscope.cancel())
                            trio.lowlevel.remove_instrument(self)

                trio.lowlevel.add_instrument(InstrumentHelper())
                await trio.sleep_forever()
            assert cscope.cancelled_caught
            watb_cscope.cancel()

        async with trio.open_nursery() as nursery:
            watb_cscope = trio.CancelScope()
            nursery.start_soon(sit_in_wait_all_tasks_blocked, watb_cscope)
            await trio.testing.wait_all_tasks_blocked()
            nursery.start_soon(get_woken_by_host_deadline, watb_cscope)

        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


@restore_unraisablehook()
def test_guest_warns_if_abandoned() -> None:
    # This warning is emitted from the garbage collector. So we have to make
    # sure that our abandoned run is garbage. The easiest way to do this is to
    # put it into a function, so that we're sure all the local state,
    # traceback frames, etc. are garbage once it returns.
    def do_abandoned_guest_run() -> None:
        async def abandoned_main(in_host: InHost) -> None:
            in_host(lambda: 1 / 0)
            while True:
                await trio.lowlevel.checkpoint()

        with pytest.raises(ZeroDivisionError):
            trivial_guest_run(abandoned_main)

    with pytest.warns(  # noqa: PT031
        RuntimeWarning,
        match="Trio guest run got abandoned",
    ):
        do_abandoned_guest_run()
        gc_collect_harder()

    # If you have problems some day figuring out what's holding onto a
    # reference to the unrolled_run generator and making this test fail,
    # then this might be useful to help track it down. (It assumes you
    # also hack start_guest_run so that it does 'global W; W =
    # weakref(unrolled_run_gen)'.)
    #
    # import gc
    # print(trio._core._run.W)
    # targets = [trio._core._run.W()]
    # for i in range(15):
    #     new_targets = []
    #     for target in targets:
    #         new_targets += gc.get_referrers(target)
    #         new_targets.remove(targets)
    #     print("#####################")
    #     print(f"depth {i}: {len(new_targets)}")
    #     print(new_targets)
    #     targets = new_targets

    with pytest.raises(RuntimeError):
        trio.current_time()


def aiotrio_run(
    trio_fn: Callable[[], Awaitable[T]],
    *,
    pass_not_threadsafe: bool = True,
    run_sync_soon_not_threadsafe: InHost | None = None,
    host_uses_signal_set_wakeup_fd: bool = False,
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> T:
    loop = asyncio.new_event_loop()

    async def aio_main() -> T:
        nonlocal run_sync_soon_not_threadsafe
        trio_done_fut: asyncio.Future[Outcome[T]] = loop.create_future()

        def trio_done_callback(main_outcome: Outcome[T]) -> None:
            print(f"trio_fn finished: {main_outcome!r}")
            trio_done_fut.set_result(main_outcome)

        if pass_not_threadsafe:
            run_sync_soon_not_threadsafe = cast("InHost", loop.call_soon)

        trio.lowlevel.start_guest_run(
            trio_fn,
            run_sync_soon_threadsafe=loop.call_soon_threadsafe,
            done_callback=trio_done_callback,
            run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe,
            host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd,
            clock=clock,
            instruments=instruments,
            restrict_keyboard_interrupt_to_checkpoints=restrict_keyboard_interrupt_to_checkpoints,
            strict_exception_groups=strict_exception_groups,
        )

        return (await trio_done_fut).unwrap()

    try:
        # can't use asyncio.run because that fails on Windows (3.10, x64, with
        # Komodia LSP)
        return loop.run_until_complete(aio_main())
    finally:
        loop.close()


def test_guest_mode_on_asyncio() -> None:
    async def trio_main() -> str:
        print("trio_main!")

        to_trio, from_aio = trio.open_memory_channel[int](float("inf"))
        from_trio: asyncio.Queue[int] = asyncio.Queue()

        aio_task = asyncio.ensure_future(aio_pingpong(from_trio, to_trio))

        # Make sure we have at least one tick where we don't need to go into
        # the thread
        await trio.lowlevel.checkpoint()

        from_trio.put_nowait(0)

        async for n in from_aio:
            print(f"trio got: {n}")
            from_trio.put_nowait(n + 1)
            if n >= 10:
                aio_task.cancel()
                return "trio-main-done"

        raise AssertionError("should never be reached")  # pragma: no cover

    async def aio_pingpong(
        from_trio: asyncio.Queue[int],
        to_trio: MemorySendChannel[int],
    ) -> None:
        print("aio_pingpong!")

        try:
            while True:
                n = await from_trio.get()
                print(f"aio got: {n}")
                to_trio.send_nowait(n + 1)
        except asyncio.CancelledError:
            raise
        except:  # pragma: no cover
            traceback.print_exc()
            raise

    assert (
        aiotrio_run(
            trio_main,
            # Not all versions of asyncio we test on can actually be trusted,
            # but this test doesn't care about signal handling, and it's
            # easier to just avoid the warnings.
            host_uses_signal_set_wakeup_fd=True,
        )
        == "trio-main-done"
    )

    assert (
        aiotrio_run(
            trio_main,
            # Also check that passing only call_soon_threadsafe works, via the
            # fallback path where we use it for everything.
            pass_not_threadsafe=False,
            host_uses_signal_set_wakeup_fd=True,
        )
        == "trio-main-done"
    )


def test_guest_mode_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    with monkeypatch.context() as m:

        async def crash_in_run_loop(in_host: InHost) -> None:
            m.setattr("trio._core._run.GLOBAL_RUN_CONTEXT.runner.runq", "HI")
            await trio.sleep(1)

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_run_loop)

    with monkeypatch.context() as m:

        async def crash_in_io(in_host: InHost) -> None:
            m.setattr("trio._core._run.TheIOManager.get_events", None)
            await trio.lowlevel.checkpoint()

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_io)

    with monkeypatch.context() as m:

        async def crash_in_worker_thread_io(in_host: InHost) -> None:
            t = threading.current_thread()
            old_get_events = trio._core._run.TheIOManager.get_events

            def bad_get_events(
                self: trio._core._run.TheIOManager,
                timeout: float,
            ) -> trio._core._run.EventResult:
                if threading.current_thread() is not t:
                    raise ValueError("oh no!")
                else:
                    return old_get_events(self, timeout)

            m.setattr("trio._core._run.TheIOManager.get_events", bad_get_events)

            await trio.sleep(1)

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_worker_thread_io)

    gc_collect_harder()


def test_guest_mode_ki() -> None:
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    # Check SIGINT in Trio func and in host func
    async def trio_main(in_host: InHost) -> None:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)

        # Host SIGINT should get injected into Trio
        in_host(partial(signal.raise_signal, signal.SIGINT))
        await trio.sleep(10)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main)
    assert excinfo.value.__context__ is None
    # Signal handler should be restored properly on exit
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    # Also check chaining in the case where KI is injected after main exits
    final_exc = KeyError("whoa")

    async def trio_main_raising(in_host: InHost) -> NoReturn:
        in_host(partial(signal.raise_signal, signal.SIGINT))
        raise final_exc

    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main_raising)
    assert excinfo.value.__context__ is final_exc

    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_guest_mode_autojump_clock_threshold_changing() -> None:
    # This is super obscure and probably no-one will ever notice, but
    # technically mutating the MockClock.autojump_threshold from the host
    # should wake up the guest, so let's test it.

    clock = trio.testing.MockClock()

    DURATION = 120

    async def trio_main(in_host: InHost) -> None:
        assert trio.current_time() == 0
        in_host(lambda: setattr(clock, "autojump_threshold", 0))
        await trio.sleep(DURATION)
        assert trio.current_time() == DURATION

    start = time.monotonic()
    trivial_guest_run(trio_main, clock=clock)
    end = time.monotonic()
    # Should be basically instantaneous, but we'll leave a generous buffer to
    # account for any CI weirdness
    assert end - start < DURATION / 2


@restore_unraisablehook()
def test_guest_mode_asyncgens() -> None:
    record = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)
            record.add((label, library))

    async def iterate_in_aio() -> None:
        await agen("asyncio").asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()

        await agen("trio").asend(None)

        gc_collect_harder()

    aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True)

    assert record == {("asyncio", "asyncio"), ("trio", "trio")}


@restore_unraisablehook()
def test_guest_mode_asyncgens_garbage_collection() -> None:
    record: set[tuple[str, str, bool]] = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        class A:
            pass

        a = A()
        a_wr = weakref.ref(a)
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)

            del a
            if sys.implementation.name == "pypy":
                gc_collect_harder()

            record.add((label, library, a_wr() is None))

    async def iterate_in_aio() -> None:
        await agen("asyncio").asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()

        await agen("trio").asend(None)

        gc_collect_harder()

    aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True)

    assert record == {("asyncio", "asyncio", True), ("trio", "trio", True)}


# ---------------------------------------------------------------------------
# Callback delivery discipline: each scheduled callback runs exactly once,
# on the host (guest) thread, in scheduling order; callbacks that arrive late
# (after the guest exited) are rejected explicitly; host pause/resume and
# guest-side failures each converge without leaving wakeup handles behind.
# ---------------------------------------------------------------------------


class _CallbackHost:
    """Minimal FIFO host loop whose schedulers can be observed or replaced."""

    def __init__(
        self,
        *,
        threadsafe: Callable[[Callable[[], object], queue.Queue[object]], None]
        | None = None,
        not_threadsafe: Callable[[Callable[[], object], queue.Queue[object]], None]
        | None = None,
    ) -> None:
        self.todo: queue.Queue[tuple[str, object]] = queue.Queue()
        self.done_outcomes: list[Outcome[object]] = []
        self.threadsafe_calls: list[Callable[[], object]] = []
        self.not_threadsafe_calls: list[Callable[[], object]] = []
        self._threadsafe_hook = threadsafe
        self._not_threadsafe_hook = not_threadsafe

    def threadsafe(self, fn: Callable[[], object]) -> None:
        self.threadsafe_calls.append(fn)
        if self._threadsafe_hook is not None:
            self._threadsafe_hook(fn, self.todo)
        else:
            self.todo.put(("run", fn))

    def not_threadsafe(self, fn: Callable[[], object]) -> None:
        self.not_threadsafe_calls.append(fn)
        if self._not_threadsafe_hook is not None:
            self._not_threadsafe_hook(fn, self.todo)
        else:
            self.todo.put(("run", fn))

    def done(self, outcome: Outcome[object]) -> None:
        self.done_outcomes.append(outcome)
        self.todo.put(("unwrap", outcome))

    def start(self, trio_fn: Callable[[InHost], Awaitable[T]]) -> None:
        trio.lowlevel.start_guest_run(
            trio_fn,
            self.not_threadsafe,
            run_sync_soon_threadsafe=self.threadsafe,
            run_sync_soon_not_threadsafe=self.not_threadsafe,
            done_callback=self.done,
        )

    def pump(self, timeout: float = 10) -> Outcome[object]:
        while True:
            op, obj = self.todo.get(timeout=timeout)
            if op == "run":
                assert not isinstance(obj, Outcome)
                obj()
            elif op == "unwrap":
                assert isinstance(obj, Outcome)
                return obj
            else:  # pragma: no cover
                raise NotImplementedError(op)


def _io_backend_is_closed(runner: object) -> bool:
    io_manager = runner.io_manager  # type: ignore[attr-defined]
    if sys.platform == "win32":
        return io_manager._iocp is None
    if hasattr(io_manager, "_epoll"):
        return io_manager._epoll.closed
    kqueue = getattr(io_manager, "_kqueue", None)
    return kqueue is None or getattr(kqueue, "closed", False)


def _assert_no_abandoned_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc_collect_harder()
    assert not [
        warning
        for warning in caught
        if "guest run got abandoned" in str(warning.message)
    ]


def _scheduled_guest_ticks(host: _CallbackHost) -> list[Callable[[], object]]:
    from trio._core._run import GuestState

    return [
        fn
        for fn in host.not_threadsafe_calls
        if getattr(fn, "__func__", None) is GuestState.guest_tick
    ]


def test_guest_threaded_callbacks_run_once_in_fifo_order() -> None:
    # A worker thread schedules a whole batch of callbacks into the guest
    # while it is parked waiting in the worker thread. Every single one has
    # to run, exactly once and in submission (FIFO) order.
    host = _CallbackHost()
    record: list[int] = []

    async def trio_main(in_host: InHost) -> str:
        token = trio.lowlevel.current_trio_token()
        finished = trio.Event()

        def submitter() -> None:
            for i in range(20):
                token.run_sync_soon(record.append, i)
            token.run_sync_soon(finished.set)

        threading.Thread(target=submitter).start()
        with trio.fail_after(10):
            await finished.wait()
        assert record == list(range(20))
        return "ok"

    host.start(trio_main)
    assert host.pump().unwrap() == "ok"


def test_guest_late_tick_after_exit_is_rejected_explicitly() -> None:
    host = _CallbackHost()

    async def trio_main(in_host: InHost) -> str:
        await trio.lowlevel.checkpoint()
        await trio.sleep(0.01)
        return "ok"

    host.start(trio_main)
    outcome = host.pump()
    assert outcome.unwrap() == "ok"
    # done_callback fires exactly once ...
    assert len(host.done_outcomes) == 1

    # ... and a stale tick the host delivers again afterwards is rejected
    # loudly instead of being silently dropped or re-running the run.
    scheduled_ticks = _scheduled_guest_ticks(host)
    assert scheduled_ticks
    with pytest.raises(trio.RunFinishedError):
        scheduled_ticks[-1]()


def test_guest_late_event_delivery_after_exit_is_rejected_explicitly() -> None:
    host = _CallbackHost()

    async def trio_main(in_host: InHost) -> str:
        # A positive timeout with nothing ready forces at least one slow
        # (worker-thread) tick, so run_sync_soon_threadsafe gets used.
        await trio.sleep(0.02)
        return "ok"

    host.start(trio_main)
    assert host.pump().unwrap() == "ok"
    assert host.threadsafe_calls
    # Simulating the host re-delivering a queued event hand-off after the
    # run finished must raise, not resurrect the run or call done twice.
    with pytest.raises(trio.RunFinishedError):
        host.threadsafe_calls[-1]()
    assert len(host.done_outcomes) == 1


def test_guest_run_sync_soon_after_exit_is_rejected() -> None:
    host = _CallbackHost()
    token_holder = []

    async def trio_main(in_host: InHost) -> None:
        token_holder.append(trio.lowlevel.current_trio_token())

    host.start(trio_main)
    host.pump().unwrap()
    with pytest.raises(trio.RunFinishedError):
        token_holder[0].run_sync_soon(lambda: None)


def test_guest_host_refusing_not_threadsafe_converges_explicitly() -> None:
    runner_holder = []

    def refuse_after_first(
        fn: Callable[[], object],
        todo: queue.Queue[tuple[str, object]],
    ) -> None:
        # Accept the initial tick so the run is initialized, then refuse
        # everything the guest tries to schedule afterwards.
        if not runner_holder:
            todo.put(("run", fn))
        else:
            raise RuntimeError("host loop is stopping")

    host = _CallbackHost(not_threadsafe=refuse_after_first)

    async def trio_main(in_host: InHost) -> None:
        runner_holder.append(trio._core._run.GLOBAL_RUN_CONTEXT.runner)
        await trio.lowlevel.checkpoint()

    host.start(trio_main)
    with pytest.raises(trio.TrioInternalError) as excinfo:
        host.pump().unwrap()
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert len(host.done_outcomes) == 1

    runner = runner_holder[0]
    assert runner.guest_terminated is True
    assert _io_backend_is_closed(runner)
    with pytest.raises(RuntimeError):
        trio.current_time()
    _assert_no_abandoned_warning()


@restore_unraisablehook()
def test_guest_host_refusing_threadsafe_converges_explicitly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The host's run_sync_soon_threadsafe refusing means its loop is gone,
    # so the worker thread cannot marshal back onto a (former) host thread.
    # Run the host on a dedicated thread, like a real GUI/event-loop thread,
    # whose run context dies with it.
    runner_holder: list[object] = []
    result: dict[str, object] = {}

    def always_refuse(
        fn: Callable[[], object],
        todo: queue.Queue[tuple[str, object]],
    ) -> None:
        raise RuntimeError("host loop is closed")

    def host_main() -> None:
        host = _CallbackHost(threadsafe=always_refuse)
        result["host"] = host

        async def trio_main(in_host: InHost) -> None:
            runner_holder.append(trio._core._run.GLOBAL_RUN_CONTEXT.runner)
            await trio.sleep(0.05)

        host.start(trio_main)
        result["outcome"] = host.pump()
        gc_collect_harder()

    host_thread = threading.Thread(target=host_main)
    host_thread.start()
    host_thread.join(10)
    assert not host_thread.is_alive()

    outcome = cast("Outcome[object]", result["outcome"])
    with pytest.raises(trio.TrioInternalError) as excinfo:
        outcome.unwrap()
    assert isinstance(excinfo.value.__cause__, RuntimeError)

    host = cast("_CallbackHost", result["host"])
    assert len(host.done_outcomes) == 1

    runner = runner_holder[0]
    assert runner.guest_terminated is True  # type: ignore[attr-defined]
    assert _io_backend_is_closed(runner)
    # The dead host thread's run context is gone, so this (other) thread is
    # left clean and could start a new run.
    with pytest.raises(RuntimeError):
        trio.current_time()
    # The lost worker delivery must not have been dumped onto stderr; it
    # was reported through done_callback instead.
    assert (
        "Exception while delivering result of thread"
        not in capsys.readouterr().err
    )


@restore_unraisablehook()
def test_start_guest_run_refusing_initial_tick_cleans_up() -> None:
    def refuse_all(
        fn: Callable[[], object],
        todo: queue.Queue[tuple[str, object]],
    ) -> None:
        raise RuntimeError("host never started")

    # Put the whole failed start in its own function scope: the host
    # exception's traceback frames anchor the primed runner, and only get
    # released once this returns (same trick as the abandoned-run test).
    def attempt() -> None:
        host = _CallbackHost(not_threadsafe=refuse_all)

        async def trio_main(in_host: InHost) -> str:
            return "ok"  # pragma: no cover

        with pytest.raises(RuntimeError, match="host never started"):
            host.start(trio_main)
        # The run never started: no done callback, context cleared.
        assert host.done_outcomes == []
        with pytest.raises(RuntimeError):
            trio.current_time()

    attempt()
    # Collect the primed-but-never-driven objects now (under the default
    # unraisable hook); abrupt teardown here is unavoidable without a
    # cooperating host, same as a genuinely abandoned run.
    gc_collect_harder()
    _assert_no_abandoned_warning()


def test_force_guest_tick_asap_after_exit_is_rejected() -> None:
    host = _CallbackHost()
    runner_holder = []

    async def trio_main(in_host: InHost) -> None:
        runner_holder.append(trio._core._run.GLOBAL_RUN_CONTEXT.runner)

    host.start(trio_main)
    host.pump().unwrap()

    runner = runner_holder[0]
    assert runner.guest_terminated is True
    # On Windows this used to raise a raw AssertionError from a closed IOCP;
    # everywhere it is now an explicit, documented rejection.
    with pytest.raises(trio.RunFinishedError):
        runner.force_guest_tick_asap()


def test_guest_paused_host_batches_callbacks_until_resume() -> None:
    # The host stops pumping callback hand-offs while a batch of callbacks
    # is scheduled from another thread, then resumes. Nothing may run (or be
    # lost) during the pause; after resume the whole batch drains in order.
    todo: queue.Queue[tuple[str, object]] = queue.Queue()
    paused = threading.Event()
    held: list[Callable[[], object]] = []
    done_outcomes: list[Outcome[object]] = []

    parking = threading.Event()
    go = threading.Event()
    submitted = threading.Event()
    record: list[int] = []

    def enqueue_threadsafe(fn: Callable[[], object]) -> None:
        if paused.is_set():
            held.append(fn)
        else:
            todo.put(("run", fn))

    def enqueue_not_threadsafe(fn: Callable[[], object]) -> None:
        todo.put(("run", fn))

    def done(outcome: Outcome[object]) -> None:
        done_outcomes.append(outcome)
        todo.put(("unwrap", outcome))

    async def trio_main(in_host: InHost) -> str:
        token = trio.lowlevel.current_trio_token()
        finished = trio.Event()

        def submitter() -> None:
            go.wait(10)
            for i in range(10):
                token.run_sync_soon(record.append, i)
            token.run_sync_soon(finished.set)
            submitted.set()

        threading.Thread(target=submitter).start()
        # Tell the host (which is pumping us right now) once this callback
        # is queued, then block with no deadline so the run parks in its
        # worker thread.
        in_host(parking.set)
        with trio.fail_after(10):
            await finished.wait()
        assert record == list(range(10))
        return "ok"

    trio.lowlevel.start_guest_run(
        trio_main,
        enqueue_not_threadsafe,
        run_sync_soon_threadsafe=enqueue_threadsafe,
        run_sync_soon_not_threadsafe=enqueue_not_threadsafe,
        done_callback=done,
    )

    # Pump until the main task has queued the "I'm parking" callback and it
    # has run; at that point the guest is blocked in the worker thread.
    while not parking.is_set():
        op, obj = todo.get(timeout=10)
        assert op == "run"
        obj()

    # Freeze thread-safe hand-offs, then release the batch submitter.
    paused.set()
    go.set()
    deadline = time.monotonic() + 10
    while not held and time.monotonic() < deadline:
        try:
            op, obj = todo.get(timeout=0.05)
        except queue.Empty:
            continue
        assert op == "run"
        obj()
    submitted.wait(10)
    assert held, "worker thread never handed off while host was paused"
    assert record == []

    # Resume: the held hand-off drains the whole batch in one go, in order.
    paused.clear()
    for fn in held:
        todo.put(("run", fn))
    while True:
        op, obj = todo.get(timeout=10)
        if op == "unwrap":
            assert isinstance(obj, Outcome)
            assert obj.unwrap() == "ok"
            break
        obj()

    assert record == list(range(10))
    assert len(done_outcomes) == 1
