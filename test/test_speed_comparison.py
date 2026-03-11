"""
Compare response latency and throughput between SGLang and R2R servers.

Setup:
  R2R:     CUDA_VISIBLE_DEVICES=0,1 python script/inference/launch_r2r_server.py --port 30000 --config-path config/Qwen3-0.6B+Qwen3-32B.yaml
  SGLang:  CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server --tp 2 --port 30001 --model-path Qwen/Qwen3-32B
Usage:
  python test/test_speed_comparison.py --num-requests 8 --rps 0.1
  python test/test_speed_comparison.py --num-requests 8 --max-batch-size 1
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import tiktoken
from datasets import load_dataset
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SERVER_NAMES = ("SGLang", "R2R")
SERVER_COLORS = {"SGLang": "blue", "R2R": "green"}
INITIAL_TIMELINE_SCALE = 60.0
DISPLAY_REFRESH_INTERVAL = 0.25
REQUEST_TIMEOUT_SECONDS = 300
TIMELINE_WIDTH = 50
RPS_STARTUP_GRACE_SECONDS = 0.25

RequestResult = Dict[str, Any]
ResultsData = List[Tuple[str, RequestResult]]
PendingRequests = Dict[str, Dict[int, float]]

console = Console()
results_lock = Lock()
pending_requests: PendingRequests = {name: {} for name in SERVER_NAMES}
timeline_scale = [INITIAL_TIMELINE_SCALE]


@dataclass(frozen=True)
class LoadMode:
    rps: Optional[float] = None
    max_concurrency: Optional[int] = None
    label: str = "burst"

    @classmethod
    def from_args(cls, rps: Optional[float], max_batch: Optional[int]) -> "LoadMode":
        if rps is not None:
            if rps <= 0:
                raise ValueError("--rps must be positive")
            return cls(rps=rps, label=f"{rps:g} RPS")
        if max_batch is not None:
            if max_batch <= 0:
                raise ValueError("--max-batch-size must be positive")
            return cls(max_concurrency=max_batch, label=f"max batch size {max_batch}")
        return cls()


def load_aime_problems(num: int) -> List[str]:
    """Load problems from AIME 1983-2024 dataset."""
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    return [ds[i]["Question"] for i in range(min(num, len(ds)))]


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def build_request_payload(prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def build_result(
    idx: int,
    start: float,
    end: float,
    t0: float,
    tokens: int = 0,
    ok: bool = True,
    err: Optional[str] = None,
) -> RequestResult:
    latency = max(end - start, 0.0)
    result: RequestResult = {
        "ok": ok,
        "idx": idx,
        "t0": start - t0,
        "t1": end - t0,
        "lat": latency,
        "tokens": tokens,
        "tps": tokens / latency if ok and latency > 0 else 0,
    }
    if err is not None:
        result["err"] = err
    return result


async def request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int,
    port: int,
    idx: int,
    enc,
    t0: float,
    name: str,
    sem: Optional[asyncio.Semaphore] = None,
) -> RequestResult:
    """Make one async request and return timing info."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = build_request_payload(prompt, max_tokens)

    async def do_request() -> RequestResult:
        start = time.time()
        with results_lock:
            pending_requests[name][idx] = start - t0
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                result = await resp.json()
                end = time.time()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = count_tokens(text, enc)
                return build_result(idx=idx, start=start, end=end, t0=t0, tokens=tokens)
        except Exception as exc:
            end = time.time()
            return build_result(idx=idx, start=start, end=end, t0=t0, ok=False, err=str(exc))

    if sem is not None:
        async with sem:
            return await do_request()
    return await do_request()


def snapshot_state(results_data: ResultsData) -> Tuple[ResultsData, PendingRequests]:
    with results_lock:
        results_copy = list(results_data)
        pending_copy = {name: dict(pending_requests[name]) for name in SERVER_NAMES}
    return results_copy, pending_copy


def reset_state() -> None:
    timeline_scale[0] = INITIAL_TIMELINE_SCALE
    with results_lock:
        for name in SERVER_NAMES:
            pending_requests[name].clear()


def update_live_display(live: Live, results_data: ResultsData, t0: float, display_rows: int) -> None:
    results_copy, pending_copy = snapshot_state(results_data)
    live.update(build_display(results_copy, t0, pending_copy, display_rows))


def record_result(
    name: str,
    result: RequestResult,
    results: ResultsData,
    live: Live,
    t0: float,
    display_rows: int,
) -> None:
    with results_lock:
        results.append((name, result))
        pending_requests[name].pop(result["idx"], None)
    update_live_display(live, results, t0, display_rows)


async def drain_completed_tasks(
    tasks: set[asyncio.Task],
    name: str,
    results: ResultsData,
    live: Live,
    t0: float,
    display_rows: int,
    timeout: Optional[float] = None,
) -> set[asyncio.Task]:
    if not tasks:
        if timeout is not None and timeout > 0:
            await asyncio.sleep(timeout)
        return tasks

    done, pending = await asyncio.wait(
        tasks,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in done:
        record_result(name, task.result(), results, live, t0, display_rows)
    return pending


async def benchmark(
    name: str,
    prompts: List[str],
    port: int,
    max_tokens: int,
    load_mode: LoadMode,
    enc,
    t0: float,
    results: ResultsData,
    live: Live,
    display_rows: int,
    schedule_start_monotonic: Optional[float] = None,
) -> None:
    """Run benchmark with either rate limiting or max concurrency control."""
    sem = asyncio.Semaphore(load_mode.max_concurrency) if load_mode.max_concurrency else None
    async with aiohttp.ClientSession() as session:
        tasks: set[asyncio.Task] = set()
        for idx, prompt in enumerate(prompts):
            if load_mode.rps is not None:
                deadline = (
                    schedule_start_monotonic + idx * (1.0 / load_mode.rps)
                    if schedule_start_monotonic is not None
                    else time.monotonic()
                )
                while True:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    remaining = await drain_completed_tasks(
                        tasks,
                        name,
                        results,
                        live,
                        t0,
                        display_rows,
                        timeout=timeout,
                    )
                    if len(remaining) == len(tasks):
                        break
                    tasks = remaining

            tasks.add(asyncio.create_task(
                request(session, prompt, max_tokens, port, idx, enc, t0, name, sem)
            ))

        while tasks:
            tasks = await drain_completed_tasks(
                tasks,
                name,
                results,
                live,
                t0,
                display_rows,
            )


def run_benchmark(
    name: str,
    prompts: List[str],
    port: int,
    max_tokens: int,
    load_mode: LoadMode,
    enc,
    t0: float,
    results: ResultsData,
    live: Live,
    display_rows: int,
    start_barrier: Optional[Barrier] = None,
    schedule_anchor: Optional[List[Optional[float]]] = None,
) -> None:
    schedule_start_monotonic = None
    if load_mode.rps is not None and start_barrier is not None and schedule_anchor is not None:
        try:
            leader = start_barrier.wait()
            if leader == 0:
                schedule_anchor[0] = time.monotonic() + RPS_STARTUP_GRACE_SECONDS
            start_barrier.wait()
            schedule_start_monotonic = schedule_anchor[0]
        except BrokenBarrierError:
            schedule_start_monotonic = time.monotonic() + RPS_STARTUP_GRACE_SECONDS

    asyncio.run(
        benchmark(
            name,
            prompts,
            port,
            max_tokens,
            load_mode,
            enc,
            t0,
            results,
            live,
            display_rows,
            schedule_start_monotonic=schedule_start_monotonic,
        )
    )


def stats(results: ResultsData) -> Optional[Dict[str, int | float]]:
    """Compute statistics."""
    ok = [r for _, r in results if r["ok"]]
    if not ok:
        return None
    lats = [r["lat"] for r in ok]
    tps = [r["tps"] for r in ok]
    return {
        "n": len(ok), "fail": len(results) - len(ok),
        "lat_avg": statistics.mean(lats), "lat_p50": statistics.median(lats),
        "tps_avg": statistics.mean(tps), "tps_med": statistics.median(tps),
        "tokens": sum(r["tokens"] for r in ok),
    }


def split_results_by_server(results_data: ResultsData) -> Dict[str, Dict[int, RequestResult]]:
    grouped: Dict[str, Dict[int, RequestResult]] = {name: {} for name in SERVER_NAMES}
    for server_name, result in results_data:
        grouped[server_name][result["idx"]] = result
    return grouped


def speedup_for_index(
    results_by_server: Dict[str, Dict[int, RequestResult]], idx: int
) -> str:
    sg_result = results_by_server["SGLang"].get(idx)
    r2r_result = results_by_server["R2R"].get(idx)
    if not sg_result or not r2r_result:
        return ""
    if not sg_result["ok"] or not r2r_result["ok"] or sg_result["tps"] <= 0:
        return ""
    return f"{r2r_result['tps'] / sg_result['tps']:.2f}x"


def build_timeline_table(
    results_by_server: Dict[str, Dict[int, RequestResult]],
    pending_by_server: PendingRequests,
    scale: float,
    current_time: float,
    display_rows: int = 10,
    width: int = TIMELINE_WIDTH,
) -> Table:
    """Build timeline table showing only latest N requests."""
    mid = scale / 2
    start_label = "0s"
    mid_label = f"{mid:.0f}s"
    end_label = f"{scale:.0f}s"
    total_label_len = len(start_label) + len(mid_label) + len(end_label)
    remaining = width - total_label_len
    left_pad = remaining // 2
    right_pad = remaining - left_pad
    scale_header = f"{start_label}{'─' * left_pad}{mid_label}{'─' * right_pad}{end_label}"

    table = Table(
        title=f"Request Timeline (Latest {display_rows}) [Scale: {scale:.0f}s]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Q#", width=4)
    table.add_column("Server", width=7)
    table.add_column(scale_header, width=width)
    table.add_column("Start", justify="right", width=6)
    table.add_column("End", justify="right", width=6)
    table.add_column("Latency", justify="right", width=7)
    table.add_column("Tok/s", justify="right", width=6)
    table.add_column("Tokens", justify="right", width=6)
    table.add_column("Speedup", justify="right", width=7)

    def bar(
        t0: float,
        t1: float,
        char: str,
        color: str,
        is_pending: bool = False,
    ) -> Text:
        if scale <= 0:
            return Text("░" * width)
        s, e = int(t0 / scale * width), int(t1 / scale * width)
        e = max(e, s + 1)
        e = min(e, width)
        txt = Text()
        txt.append("░" * s, style="dim")
        if is_pending:
            txt.append(char * (e - s), style="grey70")
        else:
            txt.append(char * (e - s), style=color)
        txt.append("░" * (width - e), style="dim")
        return txt

    def build_row(
        query_label: str,
        server_name: str,
        result: Optional[RequestResult],
        pending_start: Optional[float],
        speedup_str: str = "",
    ) -> List[object]:
        color = SERVER_COLORS[server_name]
        if result is not None:
            end_label = f"{result['t1']:.1f}" if result["ok"] else "[red]fail[/red]"
            throughput = f"{result['tps']:.0f}" if result["ok"] else "-"
            tokens = str(result["tokens"]) if result["ok"] else "-"
            latency = f"{result['lat']:.1f}"
            speedup = speedup_str if result["ok"] else ""
            return [
                query_label,
                server_name,
                bar(result["t0"], result["t1"], "█", color),
                f"{result['t0']:.1f}",
                end_label,
                latency,
                throughput,
                tokens,
                speedup,
            ]

        return [
            query_label,
            server_name,
            bar(pending_start, current_time, "█", color, is_pending=True),
            f"{pending_start:.1f}",
            "...",
            "...",
            "...",
            "...",
            "",
        ]

    all_indices = set()
    for server_name in SERVER_NAMES:
        all_indices.update(results_by_server[server_name].keys())
        all_indices.update(pending_by_server[server_name].keys())
    if not all_indices:
        return table

    sorted_indices = sorted(all_indices, reverse=True)[:display_rows]
    sorted_indices = sorted(sorted_indices)

    for i in sorted_indices:
        query_rendered = False
        speedup_str = speedup_for_index(results_by_server, i)
        for server_name in SERVER_NAMES:
            result = results_by_server[server_name].get(i)
            pending_start = pending_by_server[server_name].get(i)
            if result is None and pending_start is None:
                continue
            query_label = f"Q{i+1}" if not query_rendered else ""
            speedup = speedup_str if server_name == "R2R" else ""
            table.add_row(
                *build_row(query_label, server_name, result, pending_start, speedup)
            )
            query_rendered = True

        if i != sorted_indices[-1]:
            table.add_row("", "", "", "", "", "", "", "", "")

    return table


def build_display(
    results_data: ResultsData,
    t0: float,
    pending_by_server: Optional[PendingRequests] = None,
    display_rows: int = 10,
) -> Panel:
    """Build the live display panel."""
    results_by_server = split_results_by_server(results_data)
    current_time = time.time() - t0

    if pending_by_server is None:
        pending_by_server = {
            server_name: dict(pending_requests[server_name])
            for server_name in SERVER_NAMES
        }

    max_needed = max(
        max((r["t1"] for _, r in results_data), default=0),
        current_time,
    )

    while max_needed > timeline_scale[0]:
        timeline_scale[0] *= 2

    table = build_timeline_table(
        results_by_server,
        pending_by_server,
        timeline_scale[0],
        current_time,
        display_rows,
    )
    status_parts = []
    for server_name in SERVER_NAMES:
        server_results = results_by_server[server_name].values()
        done_count = sum(1 for result in server_results if result["ok"])
        failed_count = sum(1 for result in server_results if not result["ok"])
        status_parts.append(f"{server_name} done={done_count}, failed={failed_count}")
    title = f"[bold]Speed Comparison[/bold] ({'; '.join(status_parts)})"
    return Panel(table, title=title)


def print_summary(results_data: ResultsData) -> None:
    """Print final summary."""
    sg_stats = stats([(name, result) for name, result in results_data if name == "SGLang"])
    r2r_stats = stats([(name, result) for name, result in results_data if name == "R2R"])
    
    if not sg_stats or not r2r_stats:
        console.print("[red]Some benchmarks failed[/red]")
        return
    
    table = Table(title="Summary", show_header=True)
    table.add_column("Metric", width=25)
    table.add_column("SGLang", justify="right", width=15)
    table.add_column("R2R", justify="right", width=15)
    
    table.add_row("Successful", str(sg_stats["n"]), str(r2r_stats["n"]))
    table.add_row("Avg Latency (s)", f"{sg_stats['lat_avg']:.2f}", f"{r2r_stats['lat_avg']:.2f}")
    table.add_row("P50 Latency (s)", f"{sg_stats['lat_p50']:.2f}", f"{r2r_stats['lat_p50']:.2f}")
    table.add_row("Avg Throughput (tok/s)", f"{sg_stats['tps_avg']:.1f}", f"{r2r_stats['tps_avg']:.1f}")
    table.add_row("Total Tokens", str(sg_stats["tokens"]), str(r2r_stats["tokens"]))
    
    speedup = r2r_stats["tps_avg"] / sg_stats["tps_avg"] if sg_stats["tps_avg"] > 0 else 0
    table.add_row("[bold]Speedup[/bold]", "", f"[bold green]{speedup:.2f}x[/bold green]")
    
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SGLang vs R2R on AIME 1983-2024 problems"
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rps", type=float, default=None, help="Requests per second")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Max concurrent requests per server when --rps is unset",
    )
    parser.add_argument("--sglang-port", type=int, default=30001)
    parser.add_argument("--r2r-port", type=int, default=30000)
    parser.add_argument("--output", type=str, help="Save JSON results")
    parser.add_argument(
        "--display-rows",
        type=int,
        default=16,
        help="Number of latest requests to display",
    )
    return parser


def refresh_display_loop(
    live: Live,
    stop_event: Event,
    results_data: ResultsData,
    t0: float,
    display_rows: int,
) -> None:
    while not stop_event.wait(DISPLAY_REFRESH_INTERVAL):
        update_live_display(live, results_data, t0, display_rows)


def build_benchmark_threads(
    prompts: List[str],
    args: argparse.Namespace,
    load_mode: LoadMode,
    enc,
    t0: float,
    results_data: ResultsData,
    live: Live,
    display_rows: int,
    start_barrier: Optional[Barrier] = None,
    schedule_anchor: Optional[List[Optional[float]]] = None,
) -> List[Thread]:
    server_ports = {"SGLang": args.sglang_port, "R2R": args.r2r_port}
    return [
        Thread(
            target=run_benchmark,
            args=(
                server_name,
                prompts,
                port,
                args.max_tokens,
                load_mode,
                enc,
                t0,
                results_data,
                live,
                display_rows,
                start_barrier,
                schedule_anchor,
            ),
        )
        for server_name, port in server_ports.items()
    ]


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_mode = LoadMode.from_args(args.rps, args.max_batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    console.print(f"\n[bold]Speed Comparison: SGLang vs R2R on AIME 1983-2024[/bold]")
    console.print(f"Config: {args.num_requests} requests, {load_mode.label}, {args.max_tokens} max tokens\n")
    
    enc = tiktoken.get_encoding("cl100k_base")
    prompts = load_aime_problems(args.num_requests)
    console.print(f"✓ Loaded {len(prompts)} problems\n")

    reset_state()
    results_data: ResultsData = []
    t0 = time.time()
    display_rows = args.display_rows
    start_barrier = Barrier(len(SERVER_NAMES)) if load_mode.rps is not None else None
    schedule_anchor: Optional[List[Optional[float]]] = [None] if load_mode.rps is not None else None
    initial_panel = build_display(results_data, t0, display_rows=display_rows)
    stop_refresh = Event()

    with Live(initial_panel, console=console, refresh_per_second=4) as live:
        refresh_thread = Thread(
            target=refresh_display_loop,
            args=(live, stop_refresh, results_data, t0, display_rows),
            daemon=True,
        )
        refresh_thread.start()
        threads = build_benchmark_threads(
            prompts,
            args,
            load_mode,
            enc,
            t0,
            results_data,
            live,
            display_rows,
            start_barrier,
            schedule_anchor,
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stop_refresh.set()
        refresh_thread.join(timeout=1)
    
    console.print(f"\n✓ Completed in {time.time() - t0:.1f}s\n")
    print_summary(results_data)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "results": results_data}, f, indent=2)
        console.print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
