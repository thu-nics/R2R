"""
Reusable Gradio UI Components

This module provides factory functions for creating common UI components
that can be used across different backend demonstrations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import gradio as gr

from ..backends.base import InferenceBackend


# Default example questions for demonstrations
DEFAULT_EXAMPLES = {
    "math": """Find all positive integers n such that n² + 1 divides n³ + n² + n + 1.""",
    "reasoning": """A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?""",
    "coding": """Write a Python function to find the longest palindromic substring in a given string. Explain your approach and analyze the time complexity.""",
    "science": """Explain why the sky appears blue during the day but red/orange during sunset. Include the physics principles involved.""",
}


def create_header(
    title: str = "🛤️ Roads to Rome (R2R) Demo",
    subtitle: str = "Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing",
    description: str = "",
    logo_path: Optional[str] = None,
) -> None:
    """
    Create a header section for the demo.
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        description: Additional description markdown
        logo_path: Optional path to logo image
    """
    with gr.Row():
        if logo_path:
            with gr.Column(scale=1, min_width=100):
                gr.Image(
                    logo_path,
                    show_label=False,
                )
            with gr.Column(scale=5):
                gr.Markdown(f"""
                # {title}
                ### {subtitle}
                
                {description}
                """)
        else:
            gr.Markdown(f"""
            # {title}
            ### {subtitle}
            
            {description}
            """)


def create_model_info(
    backend: Optional[InferenceBackend] = None,
    custom_info: Optional[str] = None,
) -> None:
    """
    Create a model information display section.
    
    Args:
        backend: Backend instance to get model info from
        custom_info: Custom markdown info string (overrides backend info)
    """
    if custom_info:
        gr.Markdown(custom_info)
    elif backend:
        model_info = backend.get_model_info()
        info_lines = [f"**{key}:** {value}" for key, value in model_info.items()]
        gr.Markdown("\n".join(info_lines))


def create_input_section(
    examples: Optional[Dict[str, str]] = None,
    placeholder: str = "Type your question here or select an example above...",
    lines: int = 3,
    max_lines: int = 10,
) -> Tuple[gr.Textbox, gr.Button, gr.Button, Dict[str, gr.Button]]:
    """
    Create an input section with example buttons and text input.
    
    Args:
        examples: Dict mapping example name to question text
        placeholder: Placeholder text for input box
        lines: Number of visible lines in input box
        max_lines: Maximum lines for input box
        
    Returns:
        Tuple of (input_textbox, submit_button, clear_button, example_buttons_dict)
    """
    examples = examples or DEFAULT_EXAMPLES
    
    gr.Markdown("## 💬 Input")
    
    # Example buttons
    gr.Markdown("**Example Questions:**")
    example_buttons = {}
    
    with gr.Row():
        if "math" in examples:
            example_buttons["math"] = gr.Button("🔢 Math Problem", size="sm")
        if "reasoning" in examples:
            example_buttons["reasoning"] = gr.Button("🧠 Reasoning", size="sm")
        if "coding" in examples:
            example_buttons["coding"] = gr.Button("💻 Coding", size="sm")
        if "science" in examples:
            example_buttons["science"] = gr.Button("🔬 Science", size="sm")
        
        # Add any custom examples
        for name in examples:
            if name not in ["math", "reasoning", "coding", "science"]:
                example_buttons[name] = gr.Button(f"📝 {name.title()}", size="sm")
    
    # Input textbox
    input_textbox = gr.Textbox(
        label="Your Question",
        placeholder=placeholder,
        lines=lines,
        max_lines=max_lines,
    )
    
    # Action buttons
    with gr.Row():
        submit_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear", scale=1)
    
    return input_textbox, submit_btn, clear_btn, example_buttons


def create_color_legend(
    colors: List[Tuple[str, str, str]],
) -> None:
    """
    Create a color legend for token visualization.
    
    Args:
        colors: List of tuples (color_hex, label, description)
    """
    if not colors:
        return
    
    legend_items = []
    for color_hex, label, description in colors:
        legend_items.append(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 16px; height: 16px; border-radius: 4px; background-color: {color_hex};"></div>
                <span><strong>{label}:</strong> {description}</span>
            </div>
        """)
    
    legend_html = f"""
    <div style="display: flex; gap: 20px; padding: 10px; background: #f1f5f9; border-radius: 8px; margin-bottom: 10px;">
        {"".join(legend_items)}
    </div>
    """
    
    gr.HTML(legend_html)


def create_output_section(
    backend: Optional[InferenceBackend] = None,
    show_colored: bool = True,
    show_plain: bool = True,
    show_stats: bool = True,
    title: str = "## 📤 Output",
) -> Tuple[Optional[gr.HTML], Optional[gr.Textbox], Optional[gr.Textbox]]:
    """
    Create an output section with colored HTML, plain text, and statistics.
    
    Args:
        backend: Backend instance to determine color legend
        show_colored: Whether to show colored HTML output
        show_plain: Whether to show plain text output
        show_stats: Whether to show statistics
        title: Section title markdown
        
    Returns:
        Tuple of (colored_output, plain_output, stats_output)
    """
    gr.Markdown(title)
    
    colored_output = None
    plain_output = None
    stats_output = None
    
    # Color legend (if backend supports colored output)
    if show_colored and backend and backend.supports_colored_output():
        create_color_legend(backend.get_color_legend())
        
        colored_output = gr.HTML(
            label="Response (Colored by Model)",
        )
    
    # Plain text output
    if show_plain:
        plain_output = gr.Textbox(
            label="Plain Text Response",
            lines=10,
            max_lines=30,
            interactive=False,
        )
    
    # Statistics
    if show_stats:
        stats_output = gr.Textbox(
            label="📊 Performance Statistics",
            lines=2,
            interactive=False,
        )
    
    return colored_output, plain_output, stats_output


def create_multi_backend_output(
    backends: Dict[str, InferenceBackend],
) -> Dict[str, Tuple[Optional[gr.HTML], Optional[gr.Textbox], Optional[gr.Textbox]]]:
    """
    Create output sections for multiple backends in tabs.
    
    Args:
        backends: Dict mapping backend name to backend instance
        
    Returns:
        Dict mapping backend name to (colored_output, plain_output, stats_output) tuple
    """
    outputs = {}
    
    with gr.Tabs():
        for name, backend in backends.items():
            with gr.TabItem(backend.display_name):
                outputs[name] = create_output_section(
                    backend=backend,
                    show_colored=backend.supports_colored_output(),
                    title=f"## {backend.display_name}",
                )
    
    return outputs


def create_footer(
    paper_url: str = "https://arxiv.org/abs/2505.21600",
    project_url: str = "https://fuvty.github.io/R2R_Project_Page/",
    huggingface_url: str = "https://huggingface.co/papers/2505.21600",
    show_disclaimer: bool = True,
) -> None:
    """
    Create a footer section with links and disclaimer.
    
    Args:
        paper_url: URL to the paper
        project_url: URL to the project page
        huggingface_url: URL to HuggingFace page
        show_disclaimer: Whether to show the disclaimer
    """
    gr.Markdown("---")
    
    if show_disclaimer:
        gr.Markdown("""
        ### ⚠️ Disclaimer
        
        This demo is provided for **research purposes only** on an **"AS-IS" basis without warranties of any kind**.
        
        - Models are in experimental stages and may produce inaccurate or inconsistent outputs.
        - Generated outputs do not represent the views or opinions of the creators.
        - **Users are solely responsible** for any use of generated content.
        
        ---
        """)
    
    gr.Markdown(f"""
    **R2R** combines small and large language models by routing only critical tokens.
    [📑 Paper]({paper_url}) | [🌐 Project Page]({project_url}) | [🤗 HuggingFace]({huggingface_url})
    
    **Feel free to star our repo or cite our paper if you find it useful!** ⭐
    """)


def create_settings_panel(
    show_temperature: bool = True,
    show_top_p: bool = True,
    show_max_tokens: bool = True,
    show_threshold: bool = True,
    temperature_default: float = 0.0,
    top_p_default: float = 1.0,
    max_tokens_default: int = 4096,
    threshold_default: float = 0.5,
) -> Dict[str, gr.Component]:
    """
    Create a settings panel for generation parameters.
    
    Args:
        show_temperature: Whether to show temperature slider
        show_top_p: Whether to show top-p slider
        show_max_tokens: Whether to show max tokens slider
        show_threshold: Whether to show neural threshold slider
        temperature_default: Default temperature value
        top_p_default: Default top-p value
        max_tokens_default: Default max tokens value
        threshold_default: Default threshold value
        
    Returns:
        Dict mapping setting name to Gradio component
    """
    settings = {}
    
    with gr.Accordion("⚙️ Generation Settings", open=False):
        if show_temperature:
            settings["temperature"] = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=temperature_default,
                step=0.1,
                label="Temperature",
                info="Higher = more random, Lower = more deterministic",
            )
        
        if show_top_p:
            settings["top_p"] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=top_p_default,
                step=0.05,
                label="Top-p (Nucleus Sampling)",
                info="Cumulative probability threshold",
            )
        
        if show_max_tokens:
            settings["max_new_tokens"] = gr.Slider(
                minimum=64,
                maximum=8192,
                value=max_tokens_default,
                step=64,
                label="Max New Tokens",
                info="Maximum tokens to generate",
            )
        
        if show_threshold:
            settings["neural_threshold"] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=threshold_default,
                step=0.05,
                label="Neural Routing Threshold",
                info="Higher = more tokens to LLM, Lower = more tokens to SLM",
            )
    
    return settings


def wire_example_buttons(
    example_buttons: Dict[str, gr.Button],
    input_textbox: gr.Textbox,
    examples: Optional[Dict[str, str]] = None,
) -> None:
    """
    Wire up example buttons to populate the input textbox.
    
    Args:
        example_buttons: Dict of example name to button component
        input_textbox: Input textbox to populate
        examples: Dict of example name to question text
    """
    examples = examples or DEFAULT_EXAMPLES
    
    for name, button in example_buttons.items():
        if name in examples:
            button.click(
                fn=lambda q=examples[name]: q,
                inputs=None,
                outputs=[input_textbox],
            )
