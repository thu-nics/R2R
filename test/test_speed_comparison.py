"""
Speed Comparison: Qwen3-8B (SGLang) vs R2R (Qwen3-0.6B+Qwen3-8B)

Setup:
  R2R:     CUDA_VISIBLE_DEVICES=0,1 python script/inference/launch_r2r_server.py --port 30000 --config-path config/Qwen3-0.6B+Qwen3-32B.yaml
  SGLang:  CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server --tp 1 --port 30001 --model-path Qwen/Qwen3-32B
Usage:
  python test/speed_comparison.py --num-requests 10                    # Send all at once (burst)
  python test/speed_comparison.py --num-requests 10 --rps 2            # Rate limited
  python test/speed_comparison.py --num-requests 10 --max-batch-size 4 # Max 4 concurrent
"""

import asyncio
import aiohttp
import time
import argparse
import json
import statistics
import tiktoken
from typing import List, Dict
from datasets import load_dataset
from threading import Thread, Lock
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()
results_lock = Lock()


def load_aime_problems(num: int) -> List[str]:
    """Load problems from AIME_2024 dataset."""
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [ds[i]["Problem"] for i in range(min(num, len(ds)))]


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


async def request(session, prompt, max_tokens, port, idx, enc, t0, sem=None) -> Dict:
    """Make async request and return timing info."""
    url = f"http://0.0.0.0:{port}/v1/chat/completions"
    data = {"model": "default", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    
    async def do_request():
        start = time.time()
        try:
            async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                result = await resp.json()
                end = time.time()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = count_tokens(text, enc)
                return {"ok": True, "idx": idx, "t0": start - t0, "t1": end - t0,
                        "lat": end - start, "tokens": tokens, "tps": tokens / (end - start) if end > start else 0}
        except Exception as e:
            return {"ok": False, "idx": idx, "t0": start - t0, "t1": time.time() - t0, "lat": 0, "tokens": 0, "tps": 0, "err": str(e)}
    
    if sem:
        async with sem:
            return await do_request()
    return await do_request()


async def benchmark(name: str, prompts: List[str], port: int, max_tokens: int, rps: float, max_batch: int, enc, t0, results: List, live: Live):
    """Run benchmark. rps=rate limit, max_batch=max concurrent requests."""
    sem = asyncio.Semaphore(max_batch) if max_batch else None
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, p in enumerate(prompts):
            tasks.append(asyncio.create_task(request(session, p, max_tokens, port, i, enc, t0, sem)))
            if rps and i < len(prompts) - 1:
                await asyncio.sleep(1.0 / rps)
        
        for task in asyncio.as_completed(tasks):
            r = await task
            with results_lock:
                results.append((name, r))
                live.update(build_display(live.get_renderable().results_data))


def run_benchmark(name, prompts, port, max_tokens, rps, max_batch, enc, t0, results, live):
    asyncio.run(benchmark(name, prompts, port, max_tokens, rps, max_batch, enc, t0, results, live))


def stats(results: List[Dict]) -> Dict:
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


def build_timeline_table(sg_results: List[Dict], r2r_results: List[Dict], max_time: float, width: int = 50) -> Table:
    """Build timeline table."""
    table = Table(title="Request Timeline", show_header=True, header_style="bold")
    table.add_column("Q#", width=4)
    table.add_column("Server", width=7)
    table.add_column("Timeline", width=width)
    table.add_column("Start", justify="right", width=6)
    table.add_column("End", justify="right", width=6)
    table.add_column("Lat", justify="right", width=5)
    table.add_column("Tok/s", justify="right", width=5)
    table.add_column("Tok", justify="right", width=5)
    
    def bar(t0, t1, char, color):
        if max_time <= 0:
            return Text("░" * width)
        s, e = int(t0 / max_time * width), int(t1 / max_time * width)
        e = max(e, s + 1)
        txt = Text()
        txt.append("░" * s, style="dim")
        txt.append(char * (e - s), style=color)
        txt.append("░" * (width - e), style="dim")
        return txt
    
    sg_map = {r["idx"]: r for r in sg_results}
    r2r_map = {r["idx"]: r for r in r2r_results}
    n = max(len(sg_results), len(r2r_results), 1)
    
    for i in range(n):
        sg, r2r = sg_map.get(i), r2r_map.get(i)
        if sg:
            table.add_row(f"Q{i+1}", "SGLang", bar(sg["t0"], sg["t1"], "█", "blue"),
                          f"{sg['t0']:.1f}", f"{sg['t1']:.1f}", f"{sg['lat']:.1f}", f"{sg['tps']:.0f}", f"{sg['tokens']}")
        if r2r:
            table.add_row("", "R2R", bar(r2r["t0"], r2r["t1"], "█", "green"),
                          f"{r2r['t0']:.1f}", f"{r2r['t1']:.1f}", f"{r2r['lat']:.1f}", f"{r2r['tps']:.0f}", f"{r2r['tokens']}")
        if i < n - 1:
            table.add_row("", "", "", "", "", "", "", "")
    
    return table


def build_display(results_data: List) -> Panel:
    """Build the live display panel."""
    sg = [r for name, r in results_data if name == "SGLang" and r["ok"]]
    r2r = [r for name, r in results_data if name == "R2R" and r["ok"]]
    max_time = max((r["t1"] for _, r in results_data if r["ok"]), default=1)
    
    table = build_timeline_table(sg, r2r, max_time)
    panel = Panel(table, title=f"[bold]Speed Comparison[/bold] ({len(sg)} SGLang, {len(r2r)} R2R completed)")
    panel.results_data = results_data  # Attach data for updates
    return panel


def print_summary(results_data: List):
    """Print final summary."""
    sg_stats = stats([(n, r) for n, r in results_data if n == "SGLang"])
    r2r_stats = stats([(n, r) for n, r in results_data if n == "R2R"])
    
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


def main():
    parser = argparse.ArgumentParser(description="Compare SGLang vs R2R on AIME problems")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rps", type=float, default=1, help="Requests per second (default: all at once)")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Max concurrent requests (default: unlimited)")
    parser.add_argument("--sglang-port", type=int, default=30001)
    parser.add_argument("--r2r-port", type=int, default=30000)
    parser.add_argument("--output", type=str, help="Save JSON results")
    args = parser.parse_args()
    
    rps_str = f"{args.rps} RPS" if args.rps else "burst"
    batch_str = f", batch={args.max_batch_size}" if args.max_batch_size else ""
    console.print(f"\n[bold]Speed Comparison: SGLang vs R2R on AIME 2024[/bold]")
    console.print(f"Config: {args.num_requests} requests, {rps_str}{batch_str}, {args.max_tokens} max tokens\n")
    
    enc = tiktoken.get_encoding("cl100k_base")
    prompts = load_aime_problems(args.num_requests)
    console.print(f"✓ Loaded {len(prompts)} problems\n")
    
    results_data = []
    initial_panel = build_display(results_data)
    
    t0 = time.time()
    with Live(initial_panel, console=console, refresh_per_second=4) as live:
        t1 = Thread(target=run_benchmark, args=("SGLang", prompts, args.sglang_port, args.max_tokens, args.rps, args.max_batch_size, enc, t0, results_data, live))
        t2 = Thread(target=run_benchmark, args=("R2R", prompts, args.r2r_port, args.max_tokens, args.rps, args.max_batch_size, enc, t0, results_data, live))
        t1.start(); t2.start()
        t1.join(); t2.join()
    
    console.print(f"\n✓ Completed in {time.time() - t0:.1f}s\n")
    print_summary(results_data)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({"config": vars(args), "results": results_data}, f, indent=2)
        console.print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
