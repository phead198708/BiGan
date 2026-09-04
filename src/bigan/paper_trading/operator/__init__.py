"""Safe, recoverable paper-trading operator primitives."""

from .config import OperatorConfig, load_operator_config
from .discovery import (
    DiscoveredMarket,
    DiscoveryFilters,
    DiscoverySelection,
    MarketDiscoveryError,
    parse_gamma_markets,
    select_market_windows,
)
from .ownership import AccountOwnershipError
from .pricing_inputs import ReferencePriceSample, RollingPricingInputsProvider
from .read_model import OperatorReadRepository, OperatorState, OperatorStatus
from .resolution import FinalResolution, GammaResolutionClient, parse_gamma_resolution
from .runtime import PaperTradingOperator, stable_run_id

__all__ = [
    "AccountOwnershipError",
    "DiscoveredMarket",
    "DiscoveryFilters",
    "DiscoverySelection",
    "MarketDiscoveryError",
    "OperatorConfig",
    "load_operator_config",
    "parse_gamma_markets",
    "select_market_windows",
    "FinalResolution",
    "GammaResolutionClient",
    "OperatorReadRepository",
    "OperatorState",
    "OperatorStatus",
    "PaperTradingOperator",
    "ReferencePriceSample",
    "RollingPricingInputsProvider",
    "parse_gamma_resolution",
    "stable_run_id",
]
