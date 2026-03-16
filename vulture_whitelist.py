"""
title: Vulture whitelist for false positives.
summary: |-
  Parameters that are part of the skill interface contract
  but unused in base/protocol/no-op implementations.
"""

# BaseSkill / Skill protocol — `stage` is required by the interface
# but unused in no-op base implementations and some concrete skills.
stage  # noqa: F821
