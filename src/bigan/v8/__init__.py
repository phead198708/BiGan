"""Versioned v8 trading-system architecture modules.

Phase 0 is the data-correctness firewall. Phase 1 consumes only accepted
Phase 0 artifacts and learns pure trading policy outputs. Phase 2 consumes a
frozen accepted Phase 1.5 candidate and evaluates execution-consistent PnL.
Phase 3 optimizes a differentiable, cost-aware PnL action overlay after the
Phase 2 baseline has been established.
"""
