"""Kaggle Stem Studio: reusable weights, chunked inference and ComfyUI nodes."""
import sys

# Register inside ComfyUI only. The independent inference worker imports this
# same package without importing the server or registering HTTP routes again.
if getattr(getattr(sys.modules.get('server'), 'PromptServer', None), 'instance', None) is not None:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
