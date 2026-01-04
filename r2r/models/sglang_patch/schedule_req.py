from typing import Optional, Tuple, Union, List, Dict
from sglang.srt.sampling.sampling_params import SamplingParams

class SimpleSamplingParams:
    def __init__(self, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0, max_new_tokens: int = 128):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
    
    def derive_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

class WaitingReq:
    def __init__(
        self,
        rid: int,
        new_token_ids: List[int],
        sampling_params: Optional[SimpleSamplingParams] = None,
        status: str = "need",
    ):
        self.rid = rid
        self.new_token_ids = new_token_ids
        self.sampling_params = sampling_params
        self.status = status

