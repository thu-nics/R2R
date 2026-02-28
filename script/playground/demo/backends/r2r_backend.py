"""
R2R (Roads to Rome) Inference Backend

This backend implements dynamic token routing between a Small Language Model (SLM)
and a Large Language Model (LLM) based on reasoning divergence detection.
"""

import time
import torch
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    InferenceBackend,
    BackendConfig,
    GenerationResult,
    GenerationStatistics,
    TokenRecord,
)


class R2RBackend(InferenceBackend):
    """
    R2R inference backend with dynamic SLM-LLM token routing.
    
    This backend intelligently routes only critical, reasoning-divergent tokens
    to the large model, achieving significant speedup while maintaining accuracy.
    
    Configuration (via config.extra):
        router_path: Path to the R2R router model
        neural_threshold: Threshold for neural routing strategy (default: 0.5)
        quick_model_path: Path to the small model (optional, uses config default)
        reference_model_path: Path to the large model (optional, uses config default)
    """
    
    # Color scheme for token visualization
    SLM_COLOR = "#3B82F6"  # Blue for SLM tokens
    LLM_COLOR = "#EF4444"  # Red for LLM tokens
    
    def __init__(self, config: BackendConfig):
        """
        Initialize R2R backend.
        
        Args:
            config: Backend configuration with R2R-specific parameters in extra
        """
        super().__init__(config)
        
        # R2R-specific configuration
        self.router_path = config.get("router_path", "resource/default_router.pt")
        self.neural_threshold = config.get("neural_threshold", 0.5)
        
        # Model instances (lazy loaded)
        self._generator = None
        self._model_info_cache: Optional[Dict[str, Any]] = None
    
    @property
    def name(self) -> str:
        return "r2r"
    
    @property
    def display_name(self) -> str:
        return "🛤️ R2R (Roads to Rome)"
    
    @property
    def description(self) -> str:
        return (
            "Dynamic routing between Small Language Model (SLM) and Large Language Model (LLM). "
            "Routes only critical, reasoning-divergent tokens to the large model for speedup."
        )
    
    def load(self) -> None:
        """Load R2R models (SLM, LLM, and router)."""
        if self._is_loaded:
            return
        
        print(f"[R2R Backend] Loading models...")
        print(f"  Router: {self.router_path}")
        print(f"  Threshold: {self.neural_threshold}")
        
        try:
            # Import here to avoid circular imports and allow lazy loading
            from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
            from r2r.utils.config import MODEL_DICT
            
            strategy_kwargs = {
                'model_path': self.router_path,
                'threshold': self.neural_threshold,
            }
            
            sglang_kwargs = {
                "dtype": "bfloat16",
                "tp_size": self.config.tp_size,
            }
            
            self._generator = DynamicSimpleSGLangSelector(
                device="cuda",
                dtype=torch.bfloat16,
                switching_strategy='neural',
                strategy_kwargs=strategy_kwargs,
                is_record=False,
                sglang_kwargs=sglang_kwargs,
            )
            
            # Cache model info
            self._model_info_cache = {
                "quick_model_name": MODEL_DICT['quick']['model_name'],
                "quick_model_path": MODEL_DICT['quick']['model_path'],
                "quick_model_params": MODEL_DICT['quick']['param'],
                "reference_model_name": MODEL_DICT['reference']['model_name'],
                "reference_model_path": MODEL_DICT['reference']['model_path'],
                "reference_model_params": MODEL_DICT['reference']['param'],
                "router_path": self.router_path,
                "neural_threshold": self.neural_threshold,
            }
            
            self._is_loaded = True
            print("[R2R Backend] ✓ Models loaded successfully")
            
        except Exception as e:
            print(f"[R2R Backend] ✗ Failed to load models: {e}")
            raise RuntimeError(f"Failed to load R2R models: {e}") from e
    
    def unload(self) -> None:
        """Unload and cleanup R2R models."""
        if self._generator is not None:
            try:
                self._generator.shutdown()
            except Exception as e:
                print(f"[R2R Backend] Warning during shutdown: {e}")
            finally:
                self._generator = None
        
        self._is_loaded = False
        self._model_info_cache = None
        print("[R2R Backend] Models unloaded")
    
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate response using R2R dynamic routing.
        
        Args:
            prompt: User's input prompt/question
            
        Returns:
            GenerationResult with colored HTML output and statistics
        """
        if not self._is_loaded:
            return GenerationResult(
                error="R2R backend not loaded. Call load() first."
            )
        
        try:
            # Ensure CUDA context for quick model (GPU 1)
            self._ensure_cuda_context()
            
            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self._generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = [self._generator.tokenizer.encode(prompt_text)]
            
            start_time = time.time()
            
            # Generate with recording enabled
            generated_texts, recorders = self._generator.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=-1,
                record_generation=True,
                print_tokens=False,  # Disable terminal output
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract results
            recorder = recorders[0] if recorders else None
            generated_text = generated_texts[0] if generated_texts else ""
            
            # Build token records and HTML
            token_records = []
            html_output = ""
            statistics = GenerationStatistics(elapsed_time_s=elapsed_time)
            
            if recorder:
                # Build token records
                for record in recorder.records:
                    token_records.append(TokenRecord(
                        token_id=record.next_token,
                        token_str=record.token_str,
                        source_model=record.source_model,
                    ))
                
                # Build colored HTML
                html_output = self._build_colored_html(recorder)
                
                # Calculate statistics
                stats = recorder.get_statistics()
                statistics = GenerationStatistics(
                    total_tokens=stats['total_tokens'],
                    elapsed_time_s=elapsed_time,
                    tokens_per_second=stats['total_tokens'] / elapsed_time if elapsed_time > 0 else 0,
                    model_usage={
                        "SLM": stats['quick_model_percentage'],
                        "LLM": stats['reference_model_percentage'],
                    },
                    avg_params_billions=stats['avg_params_billions'],
                )
            else:
                # Fallback statistics
                num_tokens = len(self._generator.tokenizer.encode(generated_text))
                statistics = GenerationStatistics(
                    total_tokens=num_tokens,
                    elapsed_time_s=elapsed_time,
                    tokens_per_second=num_tokens / elapsed_time if elapsed_time > 0 else 0,
                )
            
            return GenerationResult(
                text=generated_text,
                html=html_output,
                statistics=statistics,
                token_records=token_records,
                is_complete=True,
            )
            
        except Exception as e:
            return GenerationResult(
                error=f"Generation failed: {str(e)}",
                is_complete=False,
            )
    
    def _ensure_cuda_context(self) -> None:
        """Ensure CUDA context is properly set for the quick scheduler."""
        if torch.cuda.is_available():
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            # Quick model runs on GPU 1
            torch.cuda.set_device(1)
            _ = torch.zeros(1, device='cuda:1')
            torch.cuda.synchronize(1)
    
    def _build_colored_html(self, recorder) -> str:
        """
        Build HTML with colored text based on which model generated each token.
        
        Args:
            recorder: GenerationRecorder with token records
            
        Returns:
            HTML string with colored spans
        """
        if not recorder or not recorder.records:
            return ""
        
        html_parts = []
        model_info = self.get_model_info()
        
        for record in recorder.records:
            token_str = record.token_str
            # Escape HTML special characters
            token_str = (
                token_str
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            
            if record.source_model == "quick":
                color = self.SLM_COLOR
                title = f"SLM ({model_info.get('quick_model_name', 'Small')})"
            else:
                color = self.LLM_COLOR
                title = f"LLM ({model_info.get('reference_model_name', 'Large')})"
            
            html_parts.append(
                f'<span style="color: {color};" title="{title}">{token_str}</span>'
            )
        
        return "".join(html_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded R2R models."""
        if self._model_info_cache:
            return self._model_info_cache
        
        # Return default info if not loaded
        return {
            "quick_model_name": "SLM",
            "reference_model_name": "LLM",
            "router_path": self.router_path,
            "neural_threshold": self.neural_threshold,
        }
    
    def supports_colored_output(self) -> bool:
        """R2R supports colored token output."""
        return True
    
    def get_color_legend(self) -> List[Tuple[str, str, str]]:
        """Get color legend for R2R token visualization."""
        model_info = self.get_model_info()
        return [
            (
                self.SLM_COLOR,
                "Blue",
                f"SLM ({model_info.get('quick_model_name', 'Small Language Model')})"
            ),
            (
                self.LLM_COLOR,
                "Red",
                f"LLM ({model_info.get('reference_model_name', 'Large Language Model')})"
            ),
        ]
    
    def update_threshold(self, threshold: float) -> None:
        """
        Update the neural routing threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        self.neural_threshold = threshold
        self.config.set("neural_threshold", threshold)
        
        if self._generator is not None:
            # Update the threshold in the loaded generator
            if hasattr(self._generator, 'strategy') and hasattr(self._generator.strategy, 'threshold'):
                self._generator.strategy.threshold = threshold
