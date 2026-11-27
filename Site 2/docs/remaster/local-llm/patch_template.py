"""Patch Qwen3.5's built-in chat template so it accepts system messages after the first one (Claude Code sends them).

LM Studio's API server uses the template embedded in the GGUF file, so the fix has to go there. The replacement is
the same length as the original, so only those bytes change in place. `--revert` puts the original back.

    python patch_template.py <model.gguf> [--revert]
"""
import sys

ORIGINAL = b"raise_exception('System message must be at the beginning.')"
FIXED = b"'<|im_start|>system\\n' + content + '<|im_end|>' + '\\n'"
FIXED = FIXED + b" " * (len(ORIGINAL) - len(FIXED))  # Jinja ignores the padding spaces
assert len(FIXED) == len(ORIGINAL), (len(FIXED), len(ORIGINAL))

path = sys.argv[1]
old, new = (FIXED, ORIGINAL) if "--revert" in sys.argv else (ORIGINAL, FIXED)
HEADER = 64 * 1024 * 1024  # the chat template sits in the metadata header, well inside the first 64 MB

with open(path, "r+b") as f:
    head = f.read(HEADER)
    at = head.find(old)
    if at < 0:
        state = "already patched" if head.find(new) >= 0 else "template text not found"
        sys.exit(f"nothing to do: {state}")
    if head.find(old, at + 1) >= 0:
        sys.exit("found more than once; not patching")
    f.seek(at)
    f.write(new)
print(f"{'reverted' if '--revert' in sys.argv else 'patched'} at byte {at}")
