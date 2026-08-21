import concurrent.futures as cf, time, statistics
from client import stream_chat

P = "Write a complete Python module: thread-safe LRU cache with TTL, type hints, and docstrings."
MSGS = [{"role": "user", "content": P}]

def one(port):
    return stream_chat("http://127.0.0.1:%d" % port, "qwen38", MSGS, max_tokens=300, ignore_eos=True)

for total in (16, 32):
    per = total // 2
    jobs = [8000] * per + [8001] * per
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=total) as ex:
        rs = list(ex.map(one, jobs))
    wall = time.perf_counter() - t
    toks = sum(r["out"] for r in rs)
    med = statistics.median([r["tps"] for r in rs])
    print("TWO TP=1 SERVERS, %d agents (%d+%d): agg %7.1f tok/s  per-agent %5.1f  wall %.2fs"
          % (total, per, per, toks / wall, med, wall))
