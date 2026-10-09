from __future__ import annotations

import signal
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING

import attrs

import trio

from ._util import ConflictDetector, is_main_thread

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator, Iterable
    from types import FrameType
    from typing import TypeAlias

    from typing_extensions import Self

    # A handler may be an ordinary callable, or one of signal.signal()'s
    # special sentinel values (SIG_IGN / SIG_DFL / None).
    _Handler: TypeAlias = (
        Callable[[int, FrameType | None], object] | int | signal.Handlers | None
    )

# Discussion of signal handling strategies:
#
# - On Windows signals barely exist. There are no options; signal handlers are
#   the only available API.
#
# - On Linux signalfd is arguably the natural way. Semantics: signalfd acts as
#   an *alternative* signal delivery mechanism. The way you use it is to mask
#   out the relevant signals process-wide (so that they don't get delivered
#   the normal way), and then when you read from signalfd that actually counts
#   as delivering it (despite the mask). The problem with this is that we
#   don't have any reliable way to mask out signals process-wide -- the only
#   way to do that in Python is to call pthread_sigmask from the main thread
#   *before starting any other threads*, and as a library we can't really
#   impose that, and the failure mode is annoying (signals get delivered via
#   signal handlers whether we want them to or not).
#
# - on macOS/*BSD, kqueue is the natural way. Semantics: kqueue acts as an
#   *extra* signal delivery mechanism. Signals are delivered the normal
#   way, *and* are delivered to kqueue. So you want to set them to SIG_IGN so
#   that they don't end up pending forever (I guess?). I can't find any actual
#   docs on how masking and EVFILT_SIGNAL interact. I did see someone note
#   that if a signal is pending when the kqueue filter is added then you
#   *don't* get notified of that, which makes sense. But still, we have to
#   manipulate signal state (e.g. setting SIG_IGN) which as far as Python is
#   concerned means we have to do this from the main thread.
#
# So in summary, there don't seem to be any compelling advantages to using the
# platform-native signal notification systems; they're kinda nice, but it's
# simpler to implement the naive signal-handler-based system once and be
# done. (The big advantage would be if there were a reliable way to monitor
# for SIGCHLD from outside the main thread and without interfering with other
# libraries that also want to monitor for SIGCHLD. But there isn't. I guess
# kqueue might give us that, but in kqueue we don't need it, because kqueue
# can directly monitor for child process state changes.)


# Per-signal bookkeeping for Trio's installed dispatcher.
@attrs.define(eq=False)
class _HandlerState:
    # The handler that was installed for this signal immediately before Trio
    # installed its own dispatcher. It is put back when the last Trio handler
    # for this signal is removed.
    original_handler: _Handler
    # The currently-active Trio handlers for this signal, in registration
    # order. Only the innermost (last) one receives each signal -- this
    # preserves the traditional signal.signal() save/restore stacking used by
    # nested 'with' blocks. Unlike a plain save/restore stack, though, removal
    # works no matter what order registrations finish in: when a registration
    # that is not on top goes away (e.g. its task gets cancelled while another
    # receiver is still active), the other registrations keep working.
    handlers: list[_Handler] = attrs.Factory(list)


# Signals for which Trio's dispatcher is currently installed, mapped to the
# state above. Signal handlers are process-global state and can only be
# manipulated from the main thread, so this is a plain module-level dict --
# but registrations are tracked independently and balanced, so the dict is
# always empty whenever no Trio signal handler is active (in particular, it
# does not leak across sequential calls to trio.run()).
_signal_states: dict[int, _HandlerState] = {}


def _run_special_handler(
    handler: int | signal.Handlers | None,
    signum: int,
    state: _HandlerState,
) -> None:
    if handler == signal.SIG_IGN:
        # SIG_IGN: the signal is simply discarded.
        return
    # SIG_DFL / None: hand the signal back to the operating system's default
    # disposition. This usually terminates the process, but for signals whose
    # default action is "ignore" it returns, in which case our dispatcher is
    # reinstalled so the bookkeeping stays consistent.
    signal.signal(signum, signal.SIG_DFL)
    try:
        signal.raise_signal(signum)
    finally:
        if _signal_states.get(signum) is state:
            signal.signal(signum, _signal_dispatcher)


def _signal_dispatcher(signum: int, frame: FrameType | None) -> None:
    state = _signal_states.get(signum)
    if state is None or not state.handlers:
        # This can only happen in the tiny restoration window where the
        # previous handler has already been reinstalled but this invocation of
        # the dispatcher was already pending. Re-deliver the signal to whatever
        # handler is installed now, instead of dropping it or invoking a
        # callback that has already been torn down.
        signal.raise_signal(signum)
        return
    # Only the innermost active registration receives the signal.
    handler = state.handlers[-1]
    if callable(handler):
        handler(signum, frame)
    else:
        _run_special_handler(handler, signum, state)


def _install_signal_handlers(signals: Iterable[int], handler: _Handler) -> None:
    registered: list[int] = []
    try:
        for signum in set(signals):
            state = _signal_states.get(signum)
            if state is None:
                # This may raise (e.g. for an invalid signal number); in that
                # case nothing for this signum has been recorded yet.
                original_handler = signal.signal(signum, _signal_dispatcher)
                state = _HandlerState(original_handler=original_handler)
                _signal_states[signum] = state
            state.handlers.append(handler)
            registered.append(signum)
    except BaseException:
        # Roll back everything this particular call installed, without
        # touching registrations made by anyone else.
        for signum in registered:
            _remove_signal_handlers((signum,), handler)
        raise


def _remove_signal_handlers(signals: Iterable[int], handler: _Handler) -> None:
    for signum in set(signals):
        state = _signal_states.get(signum)
        if state is None or handler not in state.handlers:
            # Duplicate removal (or this handler was never registered for this
            # signal) is a no-op. Never disturb a signal that someone else
            # still has an active registration for.
            continue
        # Drop this registration's entry, wherever it happens to be in the
        # stack. Searching from the end keeps nested registrations of the same
        # callable balanced (the innermost one is torn down first).
        for idx in range(len(state.handlers) - 1, -1, -1):
            if state.handlers[idx] == handler:
                del state.handlers[idx]
                break
        if not state.handlers:
            # Restore the previous handler *before* forgetting the state, so a
            # signal arriving in this restoration window is delivered to the
            # restored handler -- never to our dispatcher with no live
            # callbacks.
            signal.signal(signum, state.original_handler)
            del _signal_states[signum]


@contextmanager
def _signal_handler(
    signals: Iterable[int],
    handler: Callable[[int, FrameType | None], object] | int | signal.Handlers | None,
) -> Generator[None, None, None]:
    _install_signal_handlers(signals, handler)
    try:
        yield
    finally:
        _remove_signal_handlers(signals, handler)


class SignalReceiver:
    def __init__(self) -> None:
        # {signal num: None}
        self._pending: OrderedDict[int, None] = OrderedDict()
        self._lot = trio.lowlevel.ParkingLot()
        self._conflict_detector = ConflictDetector(
            "only one task can iterate on a signal receiver at a time",
        )
        self._closed = False

    def _add(self, signum: int) -> None:
        if self._closed:
            signal.raise_signal(signum)
        else:
            self._pending[signum] = None
            self._lot.unpark()

    def _redeliver_remaining(self) -> None:
        # First make sure that any signals still in the delivery pipeline will
        # get redelivered. This is idempotent, so running cleanup more than
        # once (e.g. an exception followed by a GeneratorExit) is harmless.
        self._closed = True

        # And then redeliver any that are sitting in pending. This is done
        # using a weird recursive construct to make sure we process everything
        # even if some of the handlers raise exceptions.
        def deliver_next() -> None:
            if self._pending:
                signum, _ = self._pending.popitem(last=False)
                try:
                    signal.raise_signal(signum)
                finally:
                    deliver_next()

        deliver_next()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> int:
        if self._closed:
            raise RuntimeError("open_signal_receiver block already exited")
        # In principle it would be possible to support multiple concurrent
        # calls to __anext__, but doing it without race conditions is quite
        # tricky, and there doesn't seem to be any point in trying.
        with self._conflict_detector:
            if not self._pending:
                await self._lot.park()
            else:
                await trio.lowlevel.checkpoint()
            signum, _ = self._pending.popitem(last=False)
            return signum


def get_pending_signal_count(rec: AsyncIterator[int]) -> int:
    """Helper for tests, not public or otherwise used."""
    # open_signal_receiver() always produces SignalReceiver, this should not fail.
    assert isinstance(rec, SignalReceiver)
    return len(rec._pending)


@contextmanager
def open_signal_receiver(
    *signals: signal.Signals | int,
) -> Generator[AsyncIterator[int], None, None]:
    """A context manager for catching signals.

    Entering this context manager starts listening for the given signals and
    returns an async iterator; exiting the context manager stops listening.

    The async iterator blocks until a signal arrives, and then yields it.

    Note that if you leave the ``with`` block while the iterator has
    unextracted signals still pending inside it, then they will be
    re-delivered using Python's regular signal handling logic. This avoids a
    race condition when signals arrives just before we exit the ``with``
    block.

    Args:
      signals: the signals to listen for.

    Raises:
      TypeError: if no signals were provided.

      RuntimeError: if you try to use this anywhere except Python's main
          thread. (This is a Python limitation.)

    Example:

      A common convention for Unix daemons is that they should reload their
      configuration when they receive a ``SIGHUP``. Here's a sketch of what
      that might look like using :func:`open_signal_receiver`::

         with trio.open_signal_receiver(signal.SIGHUP) as signal_aiter:
             async for signum in signal_aiter:
                 assert signum == signal.SIGHUP
                 reload_configuration()

    """
    if not signals:
        raise TypeError("No signals were provided")

    if not is_main_thread():
        raise RuntimeError(
            "Sorry, open_signal_receiver is only possible when running in "
            "Python interpreter's main thread",
        )
    token = trio.lowlevel.current_trio_token()
    queue = SignalReceiver()

    def handler(signum: int, frame: FrameType | None) -> None:
        token.run_sync_soon(queue._add, signum, idempotent=True)

    try:
        # The order here matters: exiting this block first removes our
        # callback from the shared dispatcher (and, if we were the last
        # listener, restores the previous OS-level handler). Only afterwards
        # do we mark the queue closed and re-deliver pending signals. So any
        # signal that arrives during teardown goes to a live handler -- either
        # another listener, or the handler that was there before Trio -- and
        # never to this receiver's now-invalid callback.
        with _signal_handler(signals, handler):
            yield queue
    finally:
        queue._redeliver_remaining()
