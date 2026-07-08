# =============================================================================
# utils/logging_utils.py — centralized logging setup.
#
# main() calls setup_logging() once at startup. Every other module just does
# `log = logging.getLogger(__name__)` and inherits this configuration.
# =============================================================================

import logging
import warnings


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with timestamps and silence library warnings.

    Call this once at program start (before the pipeline runs). Modules that
    only acquire a logger via getLogger(__name__) do not need to call it.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")
