"""ai_deep_analysis — URH integration package.

Imported by patched URH to render the "AI Deep Analysis" dialog and
talk to mcp-sigdetect.
"""

__version__ = "0.1.0"

from .bridge import run, save_iq_to_temp, BridgeResult, BackendChoice
