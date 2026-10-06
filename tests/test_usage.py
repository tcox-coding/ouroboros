from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from ouroboros import usage
from ouroboros.backends import CompletionError


def test_parallel_charges_belong_to_the_operation_and_survive_failure():
    @usage.scoped
    def operation():
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(copy_context().run, usage.charge, n) for n in (0.1, 0.2)]
            for f in futures:
                f.result()
        assert usage.total_or(0) == pytest.approx(0.3)
        raise CompletionError("last request failed", 0.2)
    with pytest.raises(CompletionError) as exc:
        operation()
    assert exc.value.cost_is_total and exc.value.cost_usd == pytest.approx(0.3)
    assert usage.total_or(0) == 0  # no leakage to another operation


def test_existing_session_cost_is_added_once():
    @usage.scoped
    def operation():
        usage.charge(0.02)
        assert usage.total_or(999, initial=0.3) == pytest.approx(0.32)
    operation()
