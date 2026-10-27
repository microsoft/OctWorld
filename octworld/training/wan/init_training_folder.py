"""Training-folder initialization is not included in the inference release.

The original Wan-transformer loading and training initialization code has been
removed. Load the fine-tuned checkpoint with
``octworld.models.wan_checkpoint.load_wan_checkpoint`` instead.
"""
