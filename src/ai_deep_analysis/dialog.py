"""dialog.py — Qt dialog for "AI Deep Analysis" results.

Pure-Qt; doesn't import URH internals. Can be shown from URH's signal
tab or run standalone for testing.
"""
from __future__ import annotations
import json
from typing import Optional

# URH uses PyQt6; import lazily so non-GUI code paths work
def _load_qt():
    try:
        from PyQt6 import QtWidgets, QtCore   # type: ignore
        return QtWidgets, QtCore
    except ImportError:
        from PyQt5 import QtWidgets, QtCore   # type: ignore
        return QtWidgets, QtCore


class AIDeepAnalysisDialog:
    """Wrapper that creates and shows the dialog. Instance-only — does
    NOT inherit from QDialog at import time so headless tests don't
    pull in Qt unless needed.
    """

    def __init__(self, result, parent=None):
        QtWidgets, QtCore = _load_qt()

        self._QtWidgets = QtWidgets
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("AI Deep Analysis — sigdetect")
        dlg.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Header
        hdr = QtWidgets.QLabel()
        hdr.setStyleSheet("font-weight: bold; font-size: 14px;")
        if not result.ok:
            hdr.setText(f"<span style='color:red'>Analysis FAILED via "
                        f"{result.backend.value} backend</span>")
        else:
            pl = result.pipeline
            hdr.setText(
                f"Backend: <b>{result.backend.value}</b> · "
                f"file: <code>{pl.get('file', '?')}</code> · "
                f"fs={pl.get('sample_rate_hz', '?')} · "
                f"center={pl.get('center_freq_hz', '?')} · "
                f"dur={pl.get('duration_s', '?')}s"
            )
        layout.addWidget(hdr)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # Overview tab
        ov = QtWidgets.QTextEdit()
        ov.setReadOnly(True)
        ov.setFontFamily("Menlo, Monaco, monospace")
        ov.setPlainText(self._overview_text(result))
        tabs.addTab(ov, "Overview")

        # Bursts tab
        bursts_tab = QtWidgets.QTableWidget()
        self._fill_burst_table(bursts_tab, result)
        tabs.addTab(bursts_tab, "Bursts")

        # Modulation tab
        modtab = QtWidgets.QTextEdit()
        modtab.setReadOnly(True)
        modtab.setFontFamily("Menlo, Monaco, monospace")
        modtab.setPlainText(self._modulation_text(result))
        tabs.addTab(modtab, "Modulation details")

        # Narrative tab (only if agent backend produced one)
        if result.narrative:
            nar = QtWidgets.QTextBrowser()
            nar.setMarkdown(result.narrative)
            tabs.addTab(nar, "AI narrative")

        # Raw JSON tab
        raw = QtWidgets.QPlainTextEdit()
        raw.setReadOnly(True)
        raw.setPlainText(json.dumps(
            {"ok": result.ok,
             "backend": result.backend.value,
             "pipeline": result.pipeline,
             "narrative": result.narrative,
             "error": result.error},
            indent=2, default=str))
        tabs.addTab(raw, "Raw JSON")

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        copy_json = QtWidgets.QPushButton("Copy JSON")
        copy_json.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(raw.toPlainText()))
        btns.addWidget(copy_json)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.accept)
        btns.addWidget(close)
        layout.addLayout(btns)

        self.dialog = dlg

    @staticmethod
    def _overview_text(result):
        if not result.ok:
            return f"ERROR\n\n{result.error or 'unknown'}"
        pl = result.pipeline
        lines = []
        for stage in pl.get("stages", []):
            sn = stage.get("stage")
            if sn == "find_bursts":
                lines.append(f"bursts: {stage['n_bursts']} in "
                             f"{stage['n_clusters']} clusters")
            elif sn == "identify_modulation":
                r = stage["result"]
                lines.append(f"\nmodulation: {r['modulation']} "
                             f"(confidence {r['confidence']:.2f})")
                for n in r.get("notes", []):
                    lines.append(f"  - {n}")
            elif sn == "identify_protocol":
                lines.append("\nprotocol candidates:")
                for c in stage.get("candidates", [])[:3]:
                    lines.append(f"  [{c['confidence']:.2f}] {c['name']}")
                    for n in c.get("notes", [])[:2]:
                        lines.append(f"      {n}")
        return "\n".join(lines)

    def _fill_burst_table(self, table, result):
        QtWidgets = self._QtWidgets
        bursts = []
        for stage in (result.pipeline or {}).get("stages", []):
            if stage.get("stage") == "demod_decode":
                bursts = stage.get("bursts", [])
                break
        cols = ["#", "start (s)", "duration (ms)", "mod",
                "n_pulses / n_bits", "hex"]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(bursts))
        for r, b in enumerate(bursts):
            n = b.get("n_pulses") or b.get("n_bits") or 0
            hex_val = b.get("hex_msb") or b.get("hex_after_F0") or \
                      b.get("hex_after_F0_lsb") or ""
            cells = [b.get("index", r+1),
                     b.get("start_s", ""),
                     round(b.get("end_s", 0) - b.get("start_s", 0), 4) * 1000,
                     b.get("modulation_used", ""),
                     n,
                     hex_val[:80] + ("…" if len(hex_val) > 80 else "")]
            for c, v in enumerate(cells):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        table.resizeColumnsToContents()

    @staticmethod
    def _modulation_text(result):
        if not result.ok:
            return ""
        for stage in (result.pipeline or {}).get("stages", []):
            if stage.get("stage") == "identify_modulation":
                return json.dumps(stage["result"], indent=2, default=str)
        return ""

    def exec(self):
        return self.dialog.exec()

    def show(self):
        self.dialog.show()
        return self.dialog


def show_analysis_dialog(result, parent=None):
    """Convenience: build + exec the dialog."""
    dlg = AIDeepAnalysisDialog(result, parent=parent)
    dlg.exec()
    return dlg
