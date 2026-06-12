"""tls2dseg.preprocessing — pure-function preprocessing helpers.

Phase 4 plan 05 Task 1 (ENG-07). Provides:
- ``roi``: ROI masking (pure numpy, tier_a testable)
- ``cleanup``: filtering and clustering helpers (requires pchandler, tier_b_light)

The DAG invariant (Pitfall 2 in RESEARCH.md §9) is maintained: this package
imports only ``tls2dseg.types``, numpy, scipy, sklearn — never ``tls2dseg.engines``.
"""

from __future__ import annotations
