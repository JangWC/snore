"""
Deprecated in v4.

The trained model returns:
    (logits, output_lengths)

These two values are not I and E outputs. I/E are logits[..., 0] and
logits[..., 1]. The exact parsing is implemented in pipeline.py.
"""
