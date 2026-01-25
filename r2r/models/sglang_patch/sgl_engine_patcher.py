import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import zmq
import zmq.asyncio
from PIL.Image import Image

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    MultimodalDataInputFormat,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.version import __version__
from sglang.srt.entrypoints.engine import logger, Engine

def generate(
    self,
    # The input prompt. It can be a single prompt or a batch of prompts.
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None,
    audio_data: Optional[MultimodalDataInputFormat] = None,
    video_data: Optional[MultimodalDataInputFormat] = None,
    return_logprob: Optional[Union[List[bool], bool]] = False,
    logprob_start_len: Optional[Union[List[int], int]] = None,
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
    lora_path: Optional[List[Optional[str]]] = None,
    custom_logit_processor: Optional[Union[List[str], str]] = None,
    return_hidden_states: bool = False,
    stream: bool = False,
    bootstrap_host: Optional[Union[List[str], str]] = None,
    bootstrap_port: Optional[Union[List[int], int]] = None,
    bootstrap_room: Optional[Union[List[int], int]] = None,
    data_parallel_rank: Optional[int] = None,
) -> Union[Dict, Iterator[Dict]]:
    """
    The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
    Please refer to `GenerateReqInput` for the documentation.
    """
    if self.server_args.enable_dp_attention:
        if data_parallel_rank is None:
            logger.debug("data_parallel_rank not provided, using default dispatch")
        elif data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        elif data_parallel_rank >= self.server_args.dp_size:
            raise ValueError(
                f"data_parallel_rank must be less than dp_size: {self.server_args.dp_size}"
            )

    obj = GenerateReqInput(
        text=prompt,
        input_ids=input_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
        return_logprob=return_logprob,
        logprob_start_len=logprob_start_len,
        top_logprobs_num=top_logprobs_num,
        token_ids_logprob=token_ids_logprob,
        lora_path=lora_path,
        custom_logit_processor=custom_logit_processor,
        return_hidden_states=return_hidden_states,
        stream=stream,
        bootstrap_host=bootstrap_host,
        bootstrap_port=bootstrap_port,
        bootstrap_room=bootstrap_room,
        data_parallel_rank=data_parallel_rank,
    )
    
    # patch begin
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
    # patch end

    asyncio.set_event_loop(loop)
    generator = self.tokenizer_manager.generate_request(obj, None)

    if stream:

        def generator_wrapper():
            while True:
                try:
                    chunk = loop.run_until_complete(generator.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

        return generator_wrapper()
    else:
        ret = loop.run_until_complete(generator.__anext__())
        return ret

Engine.generate = generate