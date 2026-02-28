"""
UI Components Package

This package provides reusable Gradio UI components for inference demos.
Components are designed to be backend-agnostic and work with any InferenceBackend.

Usage:
    from ui import UIBuilder, create_demo_app
    
    # Create demo with registered backends
    app = create_demo_app(
        backends=["r2r"],
        config={"router_path": "path/to/router"}
    )
    app.launch()
"""

from .builder import UIBuilder, create_demo_app
from .components import (
    create_header,
    create_input_section,
    create_output_section,
    create_footer,
    create_color_legend,
)

__all__ = [
    "UIBuilder",
    "create_demo_app",
    "create_header",
    "create_input_section",
    "create_output_section",
    "create_footer",
    "create_color_legend",
]
