"""stylo: model-agnostic stylometry evaluation harness (Phase A onward).

See prereg/phase_a_plan.md for the design this implements. The legacy
pipeline in cluster/ is frozen and not imported from here, except by the
GPT-2 arm (to come), which wraps cluster/stylometry_lib.py.
"""

__version__ = "0.1.0"
