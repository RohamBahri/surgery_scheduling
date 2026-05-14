"""Surgery scheduling research code.

The code is organized by workflow stage:

- ``core``: shared config, names, typed containers, and artifact paths.
- ``data``: loading, cleaning, temporal splitting, scope, capacity, eligibility.
- ``estimation``: duration, preference, response, profile, and bootstrap models.
- ``planning``: weekly instance construction, schedule evaluation, and audits.
- ``solvers``: deterministic weekly mixed-integer scheduling models.
- ``methods``: experiment methods with a common ``fit``/``plan`` interface.
- ``vfcg``: exact VFCG training stack used by the main learning method.
- ``experiments``: rolling-horizon orchestration.
"""
