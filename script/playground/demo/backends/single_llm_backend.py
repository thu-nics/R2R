"""
Single LLM Inference Backend (Placeholder)

This backend provides regular large language model inference without routing.
It serves as a baseline for comparison with R2R.

NOTE: This is a placeholder implementation. The actual model loading and inference
logic needs to be implemented based on your specific requirements.
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    InferenceBackend,
    BackendConfig,
    GenerationResult,
    GenerationStatistics,
    TokenRecord,
)


class SingleLLMBackend(InferenceBackend):
    """
    Single LLM inference backend (baseline).
    
    This backend uses a single large language model for all token generation,
    serving as a baseline for comparison with routing-based systems like R2R.
    
    Configuration (via config.extra):
        model_name: Display name for the model
        skip_tokenizer_init: Whether to skip tokenizer initialization in SGLang
    
    TODO: Implement actual model loading and inference logic.
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize Single LLM backend.
        
        Args:
            config: Backend configuration
        """
        super().__init__(config)
        
        # Model instances (lazy loaded)
        self._engine = None
        self._tokenizer = None
        self._model_name = config.get("model_name", "LLM")
    
    @property
    def name(self) -> str:
        return "single_llm"
    
    @property
    def display_name(self) -> str:
        return "🤖 Single LLM"
    
    @property
    def description(self) -> str:
        return (
            "Standard large language model inference. "
            "Uses a single model for all token generation (baseline for comparison)."
        )
    
    def load(self) -> None:
        """
        Load the single LLM model.
        
        TODO: Implement actual model loading logic.
        Example implementation:
            import sglang as sgl
            from transformers import AutoTokenizer
            
            self._engine = sgl.Engine(
                model_path=self.config.model_path,
                tp_size=self.config.tp_size,
                skip_tokenizer_init=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        """
        if self._is_loaded:
            return
        
        print(f"[Single LLM Backend] Loading model...")
        print(f"  Model Path: {self.config.model_path}")
        print(f"  TP Size: {self.config.tp_size}")
        
        # TODO: Implement actual model loading
        # Placeholder implementation - mark as loaded but raise on generate
        
        try:
            # Uncomment and modify this section when implementing:
            # import sglang as sgl
            # from transformers import AutoTokenizer
            #
            # self._engine = sgl.Engine(
            #     model_path=self.config.model_path,
            #     tp_size=self.config.tp_size,
            #     skip_tokenizer_init=True,
            # )
            # self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            self._is_loaded = True
            print("[Single LLM Backend] ✓ Model loaded (placeholder)")
            print("[Single LLM Backend] ⚠ NOTE: Actual inference not implemented yet")
            
        except Exception as e:
            print(f"[Single LLM Backend] ✗ Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Single LLM model: {e}") from e
    
    def unload(self) -> None:
        """Unload and cleanup model resources."""
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as e:
                print(f"[Single LLM Backend] Warning during shutdown: {e}")
            finally:
                self._engine = None
        
        self._tokenizer = None
        self._is_loaded = False
        print("[Single LLM Backend] Model unloaded")
    
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate response using single LLM.
        
        TODO: Implement actual generation logic.
        
        Args:
            prompt: User's input prompt/question
            
        Returns:
            GenerationResult with generated text and statistics
        """
        if not self._is_loaded:
            return GenerationResult(
                error="Single LLM backend not loaded. Call load() first."
            )
        
        # TODO: Implement actual generation
        # Placeholder returns an error indicating not implemented
        return GenerationResult(
            text="",
            html="",
            error="Single LLM generation not implemented yet. This is a placeholder backend.",
            is_complete=False,
        )
        
        # Example implementation (uncomment and modify when implementing):
        #
        # try:
        #     # Apply chat template
        #     messages = [{"role": "user", "content": prompt}]
        #     prompt_text = self._tokenizer.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True
        #     )
        #     
        #     start_time = time.time()
        #     
        #     # Generate using SGLang
        #     sampling_params = {
        #         "temperature": self.config.temperature,
        #         "top_p": self.config.top_p,
        #         "max_new_tokens": self.config.max_new_tokens,
        #     }
        #     
        #     result = self._engine.generate(prompt_text, sampling_params)
        #     generated_text = result.text
        #     
        #     elapsed_time = time.time() - start_time
        #     num_tokens = len(self._tokenizer.encode(generated_text))
        #     
        #     statistics = GenerationStatistics(
        #         total_tokens=num_tokens,
        #         elapsed_time_s=elapsed_time,
        #         tokens_per_second=num_tokens / elapsed_time if elapsed_time > 0 else 0,
        #         model_usage={"LLM": 100.0},
        #     )
        #     
        #     return GenerationResult(
        #         text=generated_text,
        #         html=generated_text,  # No colored output for single LLM
        #         statistics=statistics,
        #         is_complete=True,
        #     )
        #     
        # except Exception as e:
        #     return GenerationResult(
        #         error=f"Generation failed: {str(e)}",
        #         is_complete=False,
        #     )
    
    def generate_stream(
        self, prompt: str
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate response with streaming output.
        
        TODO: Implement streaming generation for better UX.
        
        Args:
            prompt: User's input prompt/question
            
        Yields:
            GenerationResult with incremental text updates
        """
        # TODO: Implement streaming generation
        # For now, fall back to non-streaming
        yield self.generate(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "model_path": self.config.model_path,
            "tp_size": self.config.tp_size,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
        }
    
    def supports_colored_output(self) -> bool:
        """Single LLM does not support colored token output."""
        return False
    
    def get_color_legend(self) -> List[Tuple[str, str, str]]:
        """No color legend for single LLM."""
        return []


class SingleLLMBackendWithR2REngine(InferenceBackend):
    """
    Single LLM backend that reuses R2R's reference model engine.
    
    This is a more practical implementation that shares the reference model
    from an existing R2R backend, avoiding duplicate model loading.
    
    Use this when you want to compare R2R with single LLM using the same
    reference model without loading it twice.
    
    Configuration (via config.extra):
        r2r_generator: Reference to an existing DynamicSimpleSGLangSelector instance
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize Single LLM backend with shared R2R engine.
        
        Args:
            config: Backend configuration with r2r_generator in extra
        """
        super().__init__(config)
        self._r2r_generator = config.get("r2r_generator")
    
    @property
    def name(self) -> str:
        return "single_llm_shared"
    
    @property
    def display_name(self) -> str:
        return "🤖 Single LLM (Shared Engine)"
    
    @property
    def description(self) -> str:
        return (
            "Single LLM inference using the reference model from R2R. "
            "Shares the same model engine to save memory."
        )
    
    def set_r2r_generator(self, generator) -> None:
        """
        Set the R2R generator to share its reference model.
        
        Args:
            generator: DynamicSimpleSGLangSelector instance from R2R backend
        """
        self._r2r_generator = generator
        self._is_loaded = generator is not None
    
    def load(self) -> None:
        """Load is handled by setting the R2R generator."""
        if self._r2r_generator is None:
            raise RuntimeError(
                "No R2R generator set. Call set_r2r_generator() first or "
                "provide r2r_generator in config.extra."
            )
        self._is_loaded = True
    
    def unload(self) -> None:
        """Unload only removes the reference, does not shutdown the shared engine."""
        self._r2r_generator = None
        self._is_loaded = False
    
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate response using only the reference model from R2R.
        
        This forces all tokens through the reference (large) model,
        effectively disabling the routing mechanism.
        
        Args:
            prompt: User's input prompt/question
            
        Returns:
            GenerationResult with generated text and statistics
        """
        if not self._is_loaded or self._r2r_generator is None:
            return GenerationResult(
                error="Shared engine backend not loaded. Set R2R generator first."
            )
        
        try:
            from sglang.srt.sampling.sampling_params import SamplingParams
            
            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self._r2r_generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self._r2r_generator.tokenizer.encode(prompt_text)
            
            # Reset cache
            self._r2r_generator.reset_cache_simple()
            self._r2r_generator.reference_prefix_indices_list = [[]]
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=-1,
                max_new_tokens=1,
                stop=[],
            )
            
            current_ids = list(input_ids)
            output_ids = []
            start_time = time.time()
            
            for step in range(self.config.max_new_tokens):
                # Get next token from reference model only
                next_tokens = self._r2r_generator.extend_step(
                    input_ids=[current_ids],
                    input_indices=[0],
                    sampling_params=sampling_params,
                )
                
                next_token = next_tokens[0]
                
                # Check for EOS
                eos_ids = getattr(self._r2r_generator.tokenizer, 'eos_token_id', None)
                if eos_ids:
                    if isinstance(eos_ids, int):
                        eos_ids = [eos_ids]
                    if next_token in eos_ids:
                        break
                
                output_ids.append(next_token)
                current_ids.append(next_token)
            
            elapsed_time = time.time() - start_time
            generated_text = self._r2r_generator.tokenizer.decode(
                output_ids, skip_special_tokens=True
            )
            
            statistics = GenerationStatistics(
                total_tokens=len(output_ids),
                elapsed_time_s=elapsed_time,
                tokens_per_second=len(output_ids) / elapsed_time if elapsed_time > 0 else 0,
                model_usage={"LLM": 100.0},
            )
            
            return GenerationResult(
                text=generated_text,
                html=generated_text,
                statistics=statistics,
                is_complete=True,
            )
            
        except Exception as e:
            return GenerationResult(
                error=f"Generation failed: {str(e)}",
                is_complete=False,
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the shared model."""
        if self._r2r_generator is not None:
            from r2r.utils.config import MODEL_DICT
            return {
                "model_name": MODEL_DICT['reference']['model_name'],
                "model_path": MODEL_DICT['reference']['model_path'],
                "model_params": MODEL_DICT['reference']['param'],
                "shared_with": "R2R Backend",
            }
        return {"status": "not_loaded"}
    
    def supports_colored_output(self) -> bool:
        """Single LLM does not support colored output."""
        return False
