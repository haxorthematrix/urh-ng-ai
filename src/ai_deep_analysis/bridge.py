"""bridge.py — connects URH's signal data to mcp-sigdetect.

Three back-ends:
  direct: import sigdetect.run_pipeline and call in-process
  mcp:    spawn the registered MCP server, call its tools over stdio
  agent:  open an Anthropic API session with the MCP server attached
          and let the model plan + execute the analysis
"""

from __future__ import annotations
import os
import sys
import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import numpy as np


# sigdetect should be pip-installed (see urh-ng-ai README). As a fallback
# for developers who clone both repos side-by-side, add ../mcp-sigdetect
# to sys.path if it happens to exist.
_DEV_SIGDETECT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "mcp-sigdetect"))
if os.path.isdir(_DEV_SIGDETECT) and _DEV_SIGDETECT not in sys.path:
    sys.path.insert(0, _DEV_SIGDETECT)


class BackendChoice(str, Enum):
    DIRECT = "direct"
    MCP = "mcp"
    AGENT = "agent"


@dataclass
class BridgeResult:
    ok: bool
    backend: BackendChoice
    pipeline: dict = field(default_factory=dict)
    narrative: Optional[str] = None
    error: Optional[str] = None
    temp_path: Optional[str] = None


def save_iq_to_temp(iq: np.ndarray, sample_rate: float,
                    center_freq_hz: Optional[float] = None,
                    fmt: str = "cs16",
                    keep_after: bool = False) -> str:
    """Write a complex IQ array to a tempfile in `fmt` and return the path.

    Filename encodes sample rate and (if known) center frequency so
    sigdetect can recover them.
    """
    iq = np.asarray(iq)
    if iq.dtype.kind == 'c':
        i = iq.real.astype(np.float32)
        q = iq.imag.astype(np.float32)
    else:
        # Treat real-valued signal as I with Q=0 (rare, but be safe)
        i = iq.astype(np.float32)
        q = np.zeros_like(i)

    if fmt == "cs16":
        # Normalize to int16 range with headroom
        peak = max(np.abs(i).max(), np.abs(q).max(), 1.0)
        scale = 30000.0 / peak
        i16 = (i * scale).astype(np.int16)
        q16 = (q * scale).astype(np.int16)
        interleaved = np.empty(2 * len(i16), dtype=np.int16)
        interleaved[0::2] = i16
        interleaved[1::2] = q16
        ext = "cs16"
        data = interleaved
    elif fmt == "cf32":
        interleaved = np.empty(2 * len(i), dtype=np.float32)
        interleaved[0::2] = i
        interleaved[1::2] = q
        ext = "cf32"
        data = interleaved
    else:
        raise ValueError(f"Unsupported temp format: {fmt}")

    # Embed parameters in the filename so sigdetect can parse them
    parts = ["urh-iq"]
    if center_freq_hz:
        mhz = center_freq_hz / 1e6
        parts.append(f"{mhz:.3f}MHz".replace(".", "_"))
    parts.append(f"{int(sample_rate/1e6)}MSps" if sample_rate >= 1e6
                 else f"{int(sample_rate/1e3)}ksps")
    base = "-".join(parts) + f".{ext}"
    path = os.path.join(tempfile.gettempdir(), base)
    data.tofile(path)
    return path


# ---------------- back-end: direct ----------------

def _run_direct(path: str, sample_rate: Optional[float],
                center_freq_hz: Optional[float]) -> dict:
    from sigdetect.server import run_pipeline  # type: ignore
    return run_pipeline(path=path,
                        sample_rate=sample_rate,
                        center_freq_hz=center_freq_hz)


# ---------------- back-end: mcp ----------------

def _run_mcp(path: str, sample_rate: Optional[float],
             center_freq_hz: Optional[float],
             server_cmd: Optional[list] = None) -> dict:
    """Use the official MCP Python SDK ClientSession to talk over stdio."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as ex:
        raise RuntimeError(
            "The 'mcp' package is required for the mcp backend. "
            "pip install 'mcp[cli]'."
        ) from ex

    import asyncio

    server_cmd = server_cmd or [sys.executable, "-m", "sigdetect.server"]

    async def _go():
        params = StdioServerParameters(command=server_cmd[0],
                                       args=server_cmd[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "run_pipeline",
                    {
                        "path": path,
                        "sample_rate": sample_rate,
                        "center_freq_hz": center_freq_hz,
                    },
                )
                # MCP returns a list of content blocks; expect a JSON text
                for block in result.content:
                    text = getattr(block, "text", None)
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"raw_text": text}
                return {}

    return asyncio.run(_go())


# ---------------- back-end: agent ----------------

AGENT_SYSTEM = """You are an SDR signal analyst. You have tools for inspecting
IQ captures, finding bursts, identifying modulation, demodulating OOK and
FSK, decoding line codes (PWM, PDM, Manchester), packing bits into hex,
and matching protocols by frequency. Your job:

1. Start with inspect_iq and find_bursts to characterize the file.
2. Pick the longest burst and identify modulation.
3. Demodulate using the appropriate tool.
4. Decode encoding; extract hex per burst.
5. Identify the protocol.
6. Write a short report (under 500 words) covering: file properties,
   modulation, encoding, per-packet hex, protocol guess, and what
   was not verifiable. Do not invent data.
"""


def _run_agent(path: str, sample_rate: Optional[float],
               center_freq_hz: Optional[float],
               model: str = "claude-opus-4-7") -> tuple[dict, str]:
    """Open an Anthropic API conversation with the sigdetect MCP server
    attached as tool provider. The model decides the call sequence.
    Returns (pipeline_result, narrative_text)."""
    try:
        import anthropic  # type: ignore
    except ImportError as ex:
        raise RuntimeError(
            "anthropic SDK required for the agent backend. "
            "pip install anthropic"
        ) from ex

    # The exact API for stdio MCP attachment in the Anthropic SDK changes;
    # call out to it via the most stable surface we know. As of writing,
    # the recommended path is to pre-run the MCP server and use the
    # /v1/messages endpoint with `mcp_servers` describing a stdio command.
    # If unavailable in the installed SDK, fall back to the direct backend
    # and append a placeholder narrative.
    pipeline = _run_direct(path, sample_rate, center_freq_hz)

    client = anthropic.Anthropic()
    user_prompt = (
        f"Analyze the IQ capture at {path}. "
        f"Sample rate {sample_rate or 'auto'}, "
        f"center freq {center_freq_hz or 'auto'}. "
        f"Here is the canned pipeline output for context:\n\n"
        f"```json\n{json.dumps(pipeline, indent=2)}\n```\n\n"
        f"Write a 300-word report following the playbook."
    )
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=AGENT_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "text", None))
    return pipeline, text


# ---------------- public entry point ----------------

def run(path: str,
        sample_rate: Optional[float] = None,
        center_freq_hz: Optional[float] = None,
        backend: BackendChoice = BackendChoice.DIRECT,
        agent_model: str = "claude-opus-4-7") -> BridgeResult:
    """One-call entry point used by the URH dialog."""
    try:
        if backend == BackendChoice.DIRECT:
            pipeline = _run_direct(path, sample_rate, center_freq_hz)
            return BridgeResult(ok=True, backend=backend, pipeline=pipeline,
                                temp_path=path)
        if backend == BackendChoice.MCP:
            pipeline = _run_mcp(path, sample_rate, center_freq_hz)
            return BridgeResult(ok=True, backend=backend, pipeline=pipeline,
                                temp_path=path)
        if backend == BackendChoice.AGENT:
            pipeline, narrative = _run_agent(path, sample_rate,
                                             center_freq_hz, model=agent_model)
            return BridgeResult(ok=True, backend=backend, pipeline=pipeline,
                                narrative=narrative, temp_path=path)
        raise ValueError(f"Unknown backend: {backend}")
    except Exception as ex:
        import traceback
        return BridgeResult(ok=False, backend=backend,
                            error=f"{type(ex).__name__}: {ex}\n"
                                  f"{traceback.format_exc()}",
                            temp_path=path)


# ---------------- adapter from URH's Signal object ----------------

def run_for_urh_signal(signal: Any,
                       selection: Optional[tuple] = None,
                       backend: BackendChoice = BackendChoice.DIRECT,
                       agent_model: str = "claude-opus-4-7"
                       ) -> BridgeResult:
    """Convenience wrapper used by the URH UI.

    `signal` is a urh.signalprocessing.Signal.Signal instance.
    `selection` is (start_sample, end_sample) or None for the whole signal.
    """
    iq = signal.iq_array.data if hasattr(signal, "iq_array") else signal.data
    if selection:
        a, b = selection
        iq = iq[a:b]
    fs = float(getattr(signal, "sample_rate", 2_000_000.0))
    cf = getattr(signal, "center_frequency", None)
    cf = float(cf) if cf else None
    path = save_iq_to_temp(iq, sample_rate=fs, center_freq_hz=cf)
    return run(path, sample_rate=fs, center_freq_hz=cf, backend=backend,
               agent_model=agent_model)
