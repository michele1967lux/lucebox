# Fork notes — read this before looking at the branches

This is a fork of [`Luce-Org/lucebox`](https://github.com/Luce-Org/lucebox),
kept only to host two branches from an independent evaluation. **It is not a
maintained alternative to upstream, and it makes no claim of priority over
upstream work.**

## Attribution, stated up front

> The linear prefix-cache livelock was independently reproduced and diagnosed
> during our testing.
>
> After implementation, we discovered that upstream
> [PR #665](https://github.com/Luce-Org/lucebox/pull/665), opened earlier, had
> independently identified essentially the same root cause and fix.
>
> Our branch is retained for reproducibility, not as a claim of priority over
> PR #665.

PR #665 is by **@jkyamog**. It was opened on 26 August 2026, two days before
our work. We did not know of it while implementing.

## The two branches

### `prod/pr665-baseline`

PR #665 applied verbatim onto `136ad9c` (clean apply), rebuilt, and used as the
production backend. **This is not our code** — authorship and design are
jkyamog's. The branch exists so the binary that produced our published
measurements is traceable to a commit.

The rebuilt binary is bit-identical (`md5 0faa4c6e…`) to the artifact used for
the numbers we published as a
[validation comment](https://github.com/Luce-Org/lucebox/pull/665#issuecomment-5458389311)
on that PR.

### `fix/linear-prefix-cache-livelock`

Our own independent implementation, written before we found #665. Kept as a
historical record, **not** as a proposed alternative.

It differs from #665 in exactly one substantive way: the fallback picks the
oldest ancestor whose cached descendants all still lie on the restore chain,
rather than the shallowest unprotected one. On a real workload the two are
indistinguishable — identical `prefix_len` sequence turn by turn, same victim,
total prefill within run-to-run noise (29.0 s vs 30.6 s). The difference only
appears when a side branch is itself pinned, and it is a policy choice about
preserving future divergence points, not a correctness issue.

We also initially justified that difference by saying it avoided "orphaning a
live side branch". That justification is **wrong** and is retracted: every
cache entry is a self-contained snapshot, so removing an ancestor invalidates
no descendant.

## What we validated, and on what

Upstream verifies #665 on NVIDIA RTX 3090. We verified it on **AMD Radeon AI
PRO R9700** (gfx1201), ROCm 7.2.1 — showing the fix is not incidentally
CUDA-specific.

| 16 turns, 4 slots, with tools | pre-fix | PR #665 |
|---|---|---|
| `prefix_len` past saturation | frozen at 3072 for 12 turns | 3072 → 7680 |
| snapshots committed | 4 | 14 |
| eviction attempts | 12 | 3 |
| total prefill | 65.2 s | 29.0 s |
| latency at turn 16 | 8.60 s, climbing | 2.45 s, flat |

The four `test_slide_*` unit tests from #665 pass on ROCm.

**One limit we declare**: a later production session did *not* exercise the
eviction path at all (49 snapshots committed, 0 evictions), so it does not
validate the fix. The only validation is the controlled 16-turn reproducer
above.

## Full write-up

Root cause analysis, measurement methodology, and the errors we made along the
way — including a reproducer that initially passed against the broken binary —
are in
[`real-world-local-llm-benchmarks`](https://github.com/michele1967lux/real-world-local-llm-benchmarks/blob/main/findings/prefix-cache/linear-livelock.md).

## License

Lucebox is Apache-2.0. This fork and its branches remain under that license.
This notes file is the only addition to `main`.
