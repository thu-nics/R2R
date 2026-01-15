# SGLang Patch Issues - Todo List

This document lists non-optimal implementations and type mismatches found in `r2r/models/sglang_patch/` compared to standard SGLang.

---

## Fixed Issues

### 1. ~~Type Mismatch: `WaitingReq.rid` should be `str`, not `int`~~ ✅ FIXED

**Location:** `r2r/models/sglang_patch/schedule_req.py:22`

Changed `rid: int` to `rid: str`

---

### 2. ~~`simple_prepare_for_extend` uses `req.rid` instead of `req.req_pool_idx`~~ ✅ FIXED

**Locations:**
- `r2r/models/sglang_patch/slm_server.py`
- `r2r/models/sglang_patch/llm_server.py`

Fixed to use `req.req_pool_idx` if set, otherwise use batch index `i`.

---

### 3. ~~Duplicate `output_ids` assignment~~ ✅ FIXED

**Location:** `r2r/models/sglang_patch/slm_server.py`

Removed duplicate assignment.

---

### 4. ~~Test file double-counting bug~~ ✅ FIXED

**Location:** `test/test_system.py:119`

Fixed token counting logic.

---

## Remaining Medium Priority Issues

### 5. Missing `device` attribute on `Req` objects

**Locations:**
- `r2r/models/sglang_patch/llm_server.py:306-307`
- `r2r/models/sglang_patch/llm_server.py:329-330`

**Current workaround:**
```python
if not hasattr(new_req, 'device'):
    new_req.device = scheduler.batch_not_need.device
```

**Options:**
1. Add `device` parameter to the patched `Req.__init__()` in `schedule_batch.py`
2. Always pass device through `ScheduleBatch` methods (proper approach)

---

### 6. Global variable `end_of_cache_loc` usage

**Locations:**
- `r2r/models/sglang_patch/slm_server.py`
- `r2r/models/sglang_patch/llm_server.py`

Global state is error-prone in multiprocessing environments.

**Recommendation:** Move to instance variable or pass as parameter.

---

## Alignment Notes

### Intentional Differences from Standard SGLang

1. **`last_cached_loc` and `last_llm_loc` fields** - Added to `Req` for KV cache coordination between SLM and LLM

2. **`WaitingReq` class** - Lightweight request wrapper for inter-model communication

3. **`SimpleSamplingParams` class** - Simplified sampling parameters for internal use

4. **`status` field on `Req`** - Custom status for disaggregation workflow (`"need"`, `"notneed"`, `"finished"`, etc.)

---

## Verification

- [x] `test_system.py` runs without type errors
- [x] Request IDs (`rid`) are consistently strings throughout
- [x] `req_pool_indices` contains integers when used for tensor creation
- [x] Generation completes successfully (533 tokens in 11.22s @ 47.52 tok/s)
