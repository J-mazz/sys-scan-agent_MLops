from typing import Any, Dict, List, Tuple

import pytest

from base_verifier import BaseVerifier


class DemoVerifier(BaseVerifier):
    def __init__(self):
        super().__init__("demo")

    def verify(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if data.get("valid", False):
            return True, []
        return False, [self._log_issue("invalid data")]


def test_base_verifier_logs_issue():
    verifier = DemoVerifier()

    assert verifier.name == "demo"
    status, issues = verifier.verify({"valid": False})
    assert status is False
    assert issues == ["[demo] invalid data"]

    status_ok, issues_ok = verifier.verify({"valid": True})
    assert status_ok is True
    assert issues_ok == []


def test_abstract_verify_enforced():
    class NoopVerifier(BaseVerifier):
        pass

    with pytest.raises(TypeError):
        NoopVerifier()  # type: ignore[abstract]
