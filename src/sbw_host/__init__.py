"""SBW (surface-brightness-weighted) host-galaxy association for Rubin LSST transients."""

__version__ = "0.2.0"

from .core import SBWResult, sbw_exponential, sbw_model, sbw_probabilities, sbw_sersic, sky_to_plane
from .stamps import sbw_stamps
from .associate import associate_host, compute_association, query_candidates, query_region, build_query
from .images import ImageAccessError, fetch_cutout, render_model_image, tangent_wcs
from .plot import method_summary, plot_association, plot_with_best_image
from .tap import get_tap_service, load_image_token, load_token

__all__ = [
    "SBWResult", "sbw_exponential", "sbw_model", "sbw_probabilities", "sbw_sersic", "sky_to_plane", "sbw_stamps",
    "associate_host", "associate_many", "compute_association", "query_candidates", "query_region", "build_query",
    "ImageAccessError", "fetch_cutout", "render_model_image", "tangent_wcs",
    "method_summary", "plot_association", "plot_with_best_image", "get_tap_service", "load_image_token", "load_token",
]


def __getattr__(name):
    # lazy, so that ``python -m sbw_host.batch`` does not import batch twice
    if name == "associate_many":
        from .batch import associate_many
        return associate_many
    raise AttributeError(name)
