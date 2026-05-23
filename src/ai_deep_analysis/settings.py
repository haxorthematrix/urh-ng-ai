"""Persistent settings for the AI Deep Analysis feature.

Stored under URH's standard settings (Qt QSettings) so they survive
restarts and respect URH's portable-mode flag.
"""

from .bridge import BackendChoice


def _qsettings():
    try:
        from urh import settings as urh_settings  # type: ignore
        return urh_settings.read_settings_class()
    except Exception:
        # Standalone fallback (testing)
        try:
            from PyQt6.QtCore import QSettings   # type: ignore
        except ImportError:
            from PyQt5.QtCore import QSettings   # type: ignore
        return QSettings("urh-ng-ai", "urh-ng-ai")


def get_backend() -> BackendChoice:
    s = _qsettings()
    raw = s.value("ai_deep_analysis/backend", BackendChoice.DIRECT.value)
    try:
        return BackendChoice(raw)
    except ValueError:
        return BackendChoice.DIRECT


def set_backend(b: BackendChoice) -> None:
    s = _qsettings()
    s.setValue("ai_deep_analysis/backend", b.value)


def get_agent_model() -> str:
    s = _qsettings()
    return s.value("ai_deep_analysis/agent_model", "claude-opus-4-7")


def set_agent_model(m: str) -> None:
    s = _qsettings()
    s.setValue("ai_deep_analysis/agent_model", m)
