"""
UI Builder - High-level Interface for Creating Gradio Demos

This module provides the UIBuilder class for constructing Gradio demos
with pluggable inference backends.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import gradio as gr

from ..backends.base import InferenceBackend, BackendConfig, GenerationResult
from ..backends.registry import BackendRegistry, get_default_registry, BackendManager
from .components import (
    create_header,
    create_model_info,
    create_input_section,
    create_output_section,
    create_footer,
    create_color_legend,
    create_settings_panel,
    wire_example_buttons,
    DEFAULT_EXAMPLES,
)


class UIBuilder:
    """
    Builder class for constructing Gradio demos with inference backends.
    
    This class provides a fluent interface for building demo applications
    that can work with any registered inference backend.
    
    Usage:
        builder = UIBuilder()
        builder.set_backend("r2r", config)
        builder.set_title("My Demo")
        builder.set_examples({"q1": "Question 1", "q2": "Question 2"})
        
        demo = builder.build()
        demo.launch()
    
    Or use the convenience function:
        demo = create_demo_app(backends=["r2r"], config={...})
    """
    
    def __init__(self, registry: Optional[BackendRegistry] = None):
        """
        Initialize the UI builder.
        
        Args:
            registry: Backend registry to use (defaults to global registry)
        """
        self.registry = registry or get_default_registry()
        
        # Configuration
        self._backends: Dict[str, InferenceBackend] = {}
        self._title = "🛤️ Roads to Rome (R2R) Demo"
        self._subtitle = "Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing"
        self._description = ""
        self._logo_path: Optional[str] = None
        self._examples: Dict[str, str] = DEFAULT_EXAMPLES.copy()
        self._show_settings = True
        self._show_disclaimer = True
        
        # Links
        self._paper_url = "https://arxiv.org/abs/2505.21600"
        self._project_url = "https://fuvty.github.io/R2R_Project_Page/"
        self._huggingface_url = "https://huggingface.co/papers/2505.21600"
    
    def add_backend(
        self, 
        name: str, 
        config: Optional[BackendConfig] = None,
        **kwargs
    ) -> "UIBuilder":
        """
        Add a backend to the demo.
        
        Args:
            name: Name of the backend type (must be registered)
            config: Backend configuration
            **kwargs: Additional config parameters (used if config is None)
            
        Returns:
            self for method chaining
        """
        if config is None:
            backend = self.registry.create_with_kwargs(name, **kwargs)
        else:
            backend = self.registry.create(name, config)
        
        self._backends[name] = backend
        return self
    
    def set_title(self, title: str, subtitle: str = "", description: str = "") -> "UIBuilder":
        """
        Set the demo title and subtitle.
        
        Args:
            title: Main title
            subtitle: Subtitle text
            description: Additional description
            
        Returns:
            self for method chaining
        """
        self._title = title
        if subtitle:
            self._subtitle = subtitle
        if description:
            self._description = description
        return self
    
    def set_logo(self, logo_path: str) -> "UIBuilder":
        """
        Set the logo image path.
        
        Args:
            logo_path: Path to logo image
            
        Returns:
            self for method chaining
        """
        self._logo_path = logo_path
        return self
    
    def set_examples(self, examples: Dict[str, str]) -> "UIBuilder":
        """
        Set example questions.
        
        Args:
            examples: Dict mapping example name to question text
            
        Returns:
            self for method chaining
        """
        self._examples = examples
        return self
    
    def add_example(self, name: str, question: str) -> "UIBuilder":
        """
        Add a single example question.
        
        Args:
            name: Example name/category
            question: The example question text
            
        Returns:
            self for method chaining
        """
        self._examples[name] = question
        return self
    
    def set_links(
        self,
        paper_url: Optional[str] = None,
        project_url: Optional[str] = None,
        huggingface_url: Optional[str] = None,
    ) -> "UIBuilder":
        """
        Set footer links.
        
        Args:
            paper_url: URL to paper
            project_url: URL to project page
            huggingface_url: URL to HuggingFace page
            
        Returns:
            self for method chaining
        """
        if paper_url:
            self._paper_url = paper_url
        if project_url:
            self._project_url = project_url
        if huggingface_url:
            self._huggingface_url = huggingface_url
        return self
    
    def show_settings(self, show: bool = True) -> "UIBuilder":
        """Enable/disable settings panel."""
        self._show_settings = show
        return self
    
    def show_disclaimer(self, show: bool = True) -> "UIBuilder":
        """Enable/disable disclaimer."""
        self._show_disclaimer = show
        return self
    
    def load_backends(self) -> "UIBuilder":
        """
        Load all added backends.
        
        Returns:
            self for method chaining
        """
        for name, backend in self._backends.items():
            print(f"[UIBuilder] Loading backend: {name}")
            backend.load()
        return self
    
    def build(self, load_backends: bool = True) -> gr.Blocks:
        """
        Build the Gradio demo application.
        
        Args:
            load_backends: Whether to load backends during build
            
        Returns:
            Gradio Blocks application
        """
        if not self._backends:
            raise ValueError("No backends added. Call add_backend() first.")
        
        if load_backends:
            self.load_backends()
        
        # Build the Gradio interface
        with gr.Blocks(title=self._title) as demo:
            self._build_header()
            gr.Markdown("---")
            self._build_model_info()
            gr.Markdown("---")
            
            # Input section
            input_box, submit_btn, clear_btn, example_btns = create_input_section(
                examples=self._examples,
            )
            
            # Settings panel
            settings = {}
            if self._show_settings:
                settings = create_settings_panel(
                    show_threshold=any(
                        b.name == "r2r" for b in self._backends.values()
                    ),
                )
            
            gr.Markdown("---")
            
            # Output section(s)
            if len(self._backends) == 1:
                # Single backend - simple output
                backend = list(self._backends.values())[0]
                colored_out, plain_out, stats_out = create_output_section(
                    backend=backend,
                    show_colored=backend.supports_colored_output(),
                    title=f"## 📤 {backend.display_name} Output",
                )
                
                # Wire up events
                self._wire_single_backend_events(
                    backend, input_box, submit_btn, clear_btn,
                    colored_out, plain_out, stats_out,
                    example_btns, settings,
                )
            else:
                # Multiple backends - tabbed output
                outputs = self._build_multi_backend_output()
                self._wire_multi_backend_events(
                    input_box, submit_btn, clear_btn,
                    outputs, example_btns, settings,
                )
            
            gr.Markdown("---")
            create_footer(
                paper_url=self._paper_url,
                project_url=self._project_url,
                huggingface_url=self._huggingface_url,
                show_disclaimer=self._show_disclaimer,
            )
        
        return demo
    
    def _build_header(self) -> None:
        """Build the header section."""
        r2r_description = ""
        if "r2r" in self._backends:
            r2r_description = """
            R2R intelligently routes **only critical, reasoning-divergent tokens to the large model**, 
            achieving significant speedup while maintaining accuracy.
            """
        
        create_header(
            title=self._title,
            subtitle=self._subtitle,
            description=self._description or r2r_description,
            logo_path=self._logo_path,
        )
    
    def _build_model_info(self) -> None:
        """Build the model information section."""
        info_parts = []
        
        for name, backend in self._backends.items():
            model_info = backend.get_model_info()
            
            if name == "r2r":
                # Special formatting for R2R
                slm_name = model_info.get("quick_model_name", "SLM")
                slm_params = model_info.get("quick_model_params", "?")
                llm_name = model_info.get("reference_model_name", "LLM")
                llm_params = model_info.get("reference_model_params", "?")
                threshold = model_info.get("neural_threshold", 0.5)
                
                info_parts.append(f"""
**{backend.display_name}:**
- 🔵 **SLM (Small):** {slm_name} ({slm_params}B params)
- 🔴 **LLM (Large):** {llm_name} ({llm_params}B params)
- 🧭 **Router Threshold:** {threshold}
                """)
            else:
                # Generic formatting
                model_name = model_info.get("model_name", "Unknown")
                info_parts.append(f"**{backend.display_name}:** {model_name}")
        
        gr.Markdown("\n".join(info_parts))
    
    def _build_multi_backend_output(self) -> Dict[str, Tuple]:
        """Build output sections for multiple backends."""
        outputs = {}
        
        with gr.Tabs():
            for name, backend in self._backends.items():
                with gr.TabItem(backend.display_name):
                    colored_out, plain_out, stats_out = create_output_section(
                        backend=backend,
                        show_colored=backend.supports_colored_output(),
                        title=f"",  # No title in tabs
                    )
                    outputs[name] = (colored_out, plain_out, stats_out)
        
        return outputs
    
    def _wire_single_backend_events(
        self,
        backend: InferenceBackend,
        input_box: gr.Textbox,
        submit_btn: gr.Button,
        clear_btn: gr.Button,
        colored_out: Optional[gr.HTML],
        plain_out: Optional[gr.Textbox],
        stats_out: Optional[gr.Textbox],
        example_btns: Dict[str, gr.Button],
        settings: Dict[str, gr.Component],
    ) -> None:
        """Wire up events for a single backend."""
        
        def generate_response(user_input: str) -> Tuple:
            if not user_input.strip():
                return ("", "", "") if colored_out else ("", "")
            
            result = backend.generate(user_input)
            
            if result.error:
                error_msg = f"Error: {result.error}"
                if colored_out:
                    return error_msg, error_msg, ""
                return error_msg, ""
            
            stats_text = f"✓ Complete | {result.statistics.format_summary()}"
            
            if colored_out:
                return result.html, result.text, stats_text
            return result.text, stats_text
        
        # Determine outputs
        if colored_out:
            outputs = [colored_out, plain_out, stats_out]
        else:
            outputs = [plain_out, stats_out]
        
        # Submit button
        submit_btn.click(
            fn=generate_response,
            inputs=[input_box],
            outputs=outputs,
        )
        
        # Enter key submit
        input_box.submit(
            fn=generate_response,
            inputs=[input_box],
            outputs=outputs,
        )
        
        # Clear button
        clear_outputs = [input_box] + [o for o in outputs if o is not None]
        clear_btn.click(
            fn=lambda: tuple([""] * len(clear_outputs)),
            inputs=None,
            outputs=clear_outputs,
        )
        
        # Example buttons
        wire_example_buttons(example_btns, input_box, self._examples)
    
    def _wire_multi_backend_events(
        self,
        input_box: gr.Textbox,
        submit_btn: gr.Button,
        clear_btn: gr.Button,
        outputs: Dict[str, Tuple],
        example_btns: Dict[str, gr.Button],
        settings: Dict[str, gr.Component],
    ) -> None:
        """Wire up events for multiple backends."""
        
        def generate_all(user_input: str) -> List:
            if not user_input.strip():
                # Return empty strings for all outputs
                result = []
                for name in self._backends:
                    colored_out, plain_out, stats_out = outputs[name]
                    if colored_out:
                        result.extend(["", "", ""])
                    else:
                        result.extend(["", ""])
                return result
            
            all_results = []
            for name, backend in self._backends.items():
                result = backend.generate(user_input)
                colored_out, plain_out, stats_out = outputs[name]
                
                if result.error:
                    error_msg = f"Error: {result.error}"
                    if colored_out:
                        all_results.extend([error_msg, error_msg, ""])
                    else:
                        all_results.extend([error_msg, ""])
                else:
                    stats_text = f"✓ Complete | {result.statistics.format_summary()}"
                    if colored_out:
                        all_results.extend([result.html, result.text, stats_text])
                    else:
                        all_results.extend([result.text, stats_text])
            
            return all_results
        
        # Flatten outputs list
        flat_outputs = []
        for name in self._backends:
            colored_out, plain_out, stats_out = outputs[name]
            if colored_out:
                flat_outputs.extend([colored_out, plain_out, stats_out])
            else:
                flat_outputs.extend([plain_out, stats_out])
        
        # Submit button
        submit_btn.click(
            fn=generate_all,
            inputs=[input_box],
            outputs=flat_outputs,
        )
        
        # Enter key submit
        input_box.submit(
            fn=generate_all,
            inputs=[input_box],
            outputs=flat_outputs,
        )
        
        # Clear button
        clear_outputs = [input_box] + flat_outputs
        clear_btn.click(
            fn=lambda: tuple([""] * len(clear_outputs)),
            inputs=None,
            outputs=clear_outputs,
        )
        
        # Example buttons
        wire_example_buttons(example_btns, input_box, self._examples)
    
    def get_backends(self) -> Dict[str, InferenceBackend]:
        """Get all added backends."""
        return self._backends
    
    def shutdown(self) -> None:
        """Shutdown all backends."""
        for name, backend in self._backends.items():
            try:
                backend.unload()
            except Exception as e:
                print(f"[UIBuilder] Error shutting down {name}: {e}")


def create_demo_app(
    backends: List[str] = ["r2r"],
    config: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    logo_path: Optional[str] = None,
    load_backends: bool = True,
) -> gr.Blocks:
    """
    Convenience function to create a demo application.
    
    Args:
        backends: List of backend names to include
        config: Configuration dict (applied to all backends)
        title: Optional custom title
        logo_path: Optional logo image path
        load_backends: Whether to load backends during creation
        
    Returns:
        Gradio Blocks application
    
    Example:
        demo = create_demo_app(
            backends=["r2r"],
            config={
                "router_path": "path/to/router.pt",
                "neural_threshold": 0.5,
                "tp_size": 2,
            },
            title="My R2R Demo",
        )
        demo.launch()
    """
    config = config or {}
    
    builder = UIBuilder()
    
    # Add backends
    for backend_name in backends:
        builder.add_backend(backend_name, **config)
    
    # Set title if provided
    if title:
        builder.set_title(title)
    
    # Set logo if provided
    if logo_path:
        builder.set_logo(logo_path)
    
    return builder.build(load_backends=load_backends)
