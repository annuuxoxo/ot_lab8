"""pytest plugin used by Track J (load with `-p tj_plugin`, this folder on PYTHONPATH).

Environment variables (all optional):
- TJ_TIMES=<file.json>   write {nodeid: {"seconds": setup+call+teardown, "outcome": ...}} at the end
- TJ_COVER=<file>        record line coverage of the `tabulate` package with one coverage context per
                         test (nodeid); lines executed outside any test (imports) get context ""
- TJ_SELECT=<file.txt>   run only the node ids listed in the file (one per line), deselect the rest

TJ_TIMES also records, per test, the top-level modules that were first imported while the test ran
(e.g. numpy, pandas). Such one-time import costs are charged to whichever test happens to run first,
so per-test times from one run are order-dependent (failure log #18).

pytest's --durations only prints 2 decimals, and most tabulate tests take < 1 ms, so we keep the
exact float durations from the test reports instead.
"""
import json
import os
import sys
from collections import defaultdict

import pytest

_times = defaultdict(float)
_outcome = {}
_new_modules = {}
_cov = None


def pytest_configure(config):
    global _cov
    path = os.environ.get("TJ_COVER")
    if path:
        import coverage
        _cov = coverage.Coverage(data_file=path, source=["tabulate"], branch=False)
        _cov.erase()
        _cov.start()          # before the test modules import tabulate


def pytest_collection_modifyitems(config, items):
    path = os.environ.get("TJ_SELECT")
    if not path:
        return
    with open(path) as f:
        keep = {line.strip() for line in f if line.strip()}
    selected = [it for it in items if it.nodeid in keep]
    deselected = [it for it in items if it.nodeid not in keep]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_setup(item):
    if _cov is not None:
        _cov.switch_context(item.nodeid)


def _top_level():
    return {name.split(".")[0] for name in list(sys.modules)}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    before = _top_level()
    yield
    _new_modules[item.nodeid] = sorted(_top_level() - before)


def pytest_runtest_logreport(report):
    _times[report.nodeid] += report.duration
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcome[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    if _cov is not None:
        _cov.switch_context("")
        _cov.stop()
        _cov.save()
    path = os.environ.get("TJ_TIMES")
    if path:
        with open(path, "w") as f:
            json.dump({k: {"seconds": v, "outcome": _outcome.get(k, "unknown"),
                        "new_modules": _new_modules.get(k, [])} for k, v in _times.items()}, f)
