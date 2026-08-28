"""Regression test: the snapshot frontier must keep advancing in a linear chat.

`test_multi_turn_prefix_cache.py` runs five turns against four slots, so it
only ever reaches the moment the cache fills. This test runs well past that
point, which is where the interesting failure lives: once the cache holds a
full linear chain A < B < C < D, the deepest entry is the only one that is not
a strict prefix of another, so the prefix-aware policy picks it as the eviction
victim -- but it is also the entry the request restores from, so the server
cancels the reservation rather than overwrite KV it is still reading. No deeper
snapshot is ever committed and the restored prefix_len freezes. Observed in a
real agent session: 62 of 91 turns pinned at prefix_len=35328, with 91 % of
their wall-clock spent re-prefilling.

Asserts, over the turns after the cache saturates:

  - prefix_len never goes backwards, and
  - it advances at least once (a frozen frontier is exactly a flat line).

Scope: this invariant holds because the prompts here are built as strict
extensions of one another. It is NOT a universal server invariant -- a changed
chat template, tool set, or system prompt, or a genuine conversation branch,
may legitimately shrink the reusable prefix.

Run against a spawned server (default), or an already-running one:
    python3 server/scripts/test_linear_prefix_cache_advances.py
    python3 server/scripts/test_linear_prefix_cache_advances.py --url http://localhost:8000
"""
import argparse
import atexit
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOG = "/tmp/test_linear_pc_server.log"


def parse_args():
    ap = argparse.ArgumentParser(description="Linear prefix-cache progression test")
    ap.add_argument("--url", type=str, default=None,
                    help="Base URL of a running dflash_server (skips spawn)")
    ap.add_argument("--server-bin", type=str, default=str(ROOT / "build-hip/dflash_server"),
                    help="Server binary to spawn (lets an A/B compare two builds)")
    ap.add_argument("--target", type=str, default=None, help="Target model GGUF")
    ap.add_argument("--draft", type=str, default=None, help="Draft model path")
    ap.add_argument("--slots", type=int, default=4, help="--prefix-cache-slots")
    ap.add_argument("--turns", type=int, default=16,
                    help="Conversation turns; must exceed --slots by a few")
    ap.add_argument("--filler-words", type=int, default=400,
                    help="Filler words per turn. Each turn must advance the "
                         "chat boundaries enough to earn its own snapshot, or "
                         "the cache never fills and the saturated regime -- "
                         "the only place the livelock lives -- is never tested")
    ap.add_argument("--tools", action="store_true",
                    help="Send a tool schema, as an agent loop does: this makes "
                         "the server pin the system+tools head as a protected "
                         "entry, the shape seen in production")
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=18183)
    ap.add_argument("--log", type=str, default=DEFAULT_LOG)
    return ap.parse_args()


def spawn_server(args):
    """Spawn a local dflash_server and return its base URL."""
    server_bin = Path(args.server_bin)
    if not server_bin.exists():
        print(f"SKIP: server binary missing ({server_bin})")
        sys.exit(0)
    if not args.target or not Path(args.target).exists():
        print(f"SKIP: target model missing (--target {args.target})")
        sys.exit(0)

    cmd = [str(server_bin), args.target,
           "--target-device", "hip:0",
           "--max-ctx", str(args.max_ctx),
           "--port", str(args.port),
           "--prefix-cache-slots", str(args.slots)]
    if args.draft:
        cmd += ["--draft", args.draft, "--draft-device", "hip:0"]

    log = open(args.log, "w")
    print(f"Spawning: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, bufsize=1)

    def cleanup():
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)

    base_url = f"http://127.0.0.1:{args.port}"
    print("Waiting for server...", flush=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"SERVER DIED; see {args.log}")
            sys.exit(2)
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=1).read()
            print("Server up.", flush=True)
            return base_url
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(2)

    print(f"Server didn't come up within 600s; see {args.log}")
    sys.exit(2)


def restored_prefix_lens(log_path):
    """prefix_len restored by each completed request, in order.

    A turn that restored nothing reports 0, so the returned list has one entry
    per completed request and stays aligned with the turn sequence.
    """
    lens = []
    try:
        with open(log_path) as f:
            for ln in f:
                if "[server] chat DONE" not in ln:
                    continue
                m = re.search(r"prefix_len=(\d+)", ln)
                lens.append(int(m.group(1)) if m else 0)
    except FileNotFoundError:
        pass
    return lens


def main():
    args = parse_args()
    if args.turns <= args.slots + 2:
        print(f"--turns ({args.turns}) must exceed --slots ({args.slots}) by at "
              f"least 3 for the saturated regime to be observable")
        sys.exit(2)

    spawned = args.url is None
    base_url = args.url if args.url else spawn_server(args)

    # A system prompt big enough that a lost prefix is visible in the latency.
    system = "You are a careful assistant. Answer in one short sentence. " * 120

    def chat_post(payload):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=900)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    history = []

    tool_schema = [{
        "type": "function",
        "function": {
            "name": f"lookup_item_{i}",
            "description": "Look up an item by name and return its details.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    } for i in range(8)]

    def turn(user):
        history.append({"role": "user", "content": user})
        payload = {
            "model": "luce-dflash",
            "messages": [{"role": "system", "content": system}, *history],
            "max_tokens": 8,
            "stream": False,
        }
        if args.tools:
            payload["tools"] = tool_schema
        t0 = time.time()
        reply = chat_post(payload)
        dt = time.time() - t0
        # Keep every turn a strict extension of the previous prompt.
        history.append({"role": "assistant", "content": reply})
        return dt, reply

    # Each question carries its own filler so the conversation grows steadily.
    latencies = []
    for i in range(1, args.turns + 1):
        filler = f"(context block {i}: " + ("detail " * args.filler_words) + ")"
        dt, reply = turn(f"Q{i}: {filler} In one word, what is item {i}?")
        latencies.append(dt)
        print(f"  turn {i:2d}: latency={dt:6.2f}s reply={reply!r:.40}", flush=True)

    if not spawned:
        print("\nNo server log to inspect (--url given); latency only.")
        print(f"latencies: {['%.2f' % t for t in latencies]}")
        return 0

    lens = restored_prefix_lens(args.log)
    print(f"\n=== restored prefix_len per turn (from {args.log}) ===")
    if len(lens) != args.turns:
        print(f"  WARNING: {len(lens)} completed requests logged for "
              f"{args.turns} turns")
    for i, v in enumerate(lens, start=1):
        print(f"  turn {i:2d}: prefix_len={v}")

    # The cache is full once `slots` snapshots exist; examine the turns after.
    first = args.slots
    tail = lens[first:]
    if len(tail) < 3:
        print(f"\nFAIL: only {len(tail)} turns observed past saturation")
        return 1

    regressions = [(first + i, tail[i - 1], tail[i])
                   for i in range(1, len(tail)) if tail[i] < tail[i - 1]]
    advanced = max(tail) > tail[0]

    print("\n=== Verdict ===")
    print(f"  turns examined (past saturation): {first + 1}..{len(lens)}")
    print(f"  prefix_len over those turns: {tail}")

    monotonic_ok = not regressions
    print(f"  never goes backwards: {'OK' if monotonic_ok else 'FAIL'}")
    for turn_idx, prev, cur in regressions:
        print(f"    turn {turn_idx + 1}: {prev} -> {cur}")

    print(f"  advances at least once: {'OK' if advanced else 'FAIL'}")
    if not advanced:
        print(f"    frontier frozen at {tail[0]} for all "
              f"{len(tail)} turns past saturation -- this is the livelock")

    ok = monotonic_ok and advanced
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
