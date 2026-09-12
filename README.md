# TiinyBench

An independent benchmark suite for the **Tiiny AI Pocket Lab**.

It measures the things a spec sheet does not: how prefill scales with prompt length,
whether throughput holds over a long generation, what actually happens when more than one
person uses the box at once, and what a reasoning model charges for the tokens nobody
reads. Every number is recorded next to the NPU utilisation and memory it was taken under,
because tokens per second without the load context is not evidence.

It produces a self-contained HTML report. No CDN, no webfonts, no JavaScript needed to read
it. One file you can email.

```bash
export TIINY_HOST=192.168.1.50          # your device, host only
export TIINY_KEY=<your device api key>  # Settings → API in TiinyOS

./tiiny-bench --catalog                 # what is installed
./tiiny-bench --label first-run         # benchmark whatever is loaded
./tiiny-bench --report                  # build report.html
```

Python standard library only. Nothing to install.

---

## What it measures

| Test | The question it answers |
|---|---|
| **Prefill scaling** | What does a long prompt cost to read? Four prompt lengths from ~70 to ~6000 tokens, each asking for a 4-token answer so decode barely registers. |
| **Sustained generation** | Does throughput hold? One unbroken 1500-token generation, with NPU utilisation sampled on a thread *while it runs*. |
| **Concurrency** | 1, 2, 4 and 8 identical requests fired at once. Aggregate throughput against per-stream throughput. |
| **Reasoning cost** | The same question with thinking off and then on. The ratio is wall time, because that is what a person waits. |

**What it does not measure: quality.** Nothing here says a model is good, only how fast it
is. A fast wrong answer is still wrong.

---

## It does not touch your box unless you ask

By default the suite benchmarks **whatever model is already running** and loads, unloads and
deletes nothing. That is the whole default behaviour and it is deliberate: a benchmark that
rearranges your device is a benchmark you cannot run on a box doing real work.

Two flags change that, and both say so before they start:

```bash
./tiiny-bench --model Qwen/Qwen3-8B --label single   # loads it, benchmarks it, unloads it
./tiiny-bench --all --label sweep                    # every text model, one at a time
```

A sweep restores whatever was loaded when it started. **Results are written after every
model**, so an hour-long sweep that gets interrupted at minute fifty keeps everything it
already measured.

---

## Reading the results

```bash
./tiiny-bench --report        # rebuilds report.html from every file in bench-results/
open report.html
```

The report reads every result file in `bench-results/`, including older ones in the
previous format, and lays them out newest first with a comparison across runs at the top.

Raw results are plain JSON, one file per run. `--show <file>` prints one.

---

## Two findings from the box this was written on

Both of these are in the sample results, and both are the kind of thing a spec sheet will
never tell you.

**Long prompts get cheaper, not more expensive.** Prefill climbed from 135 tok/s on a
70-token prompt to 755 tok/s on a 6000-token one. The per-token cost of context falls by
more than five times as the prompt grows, which is the opposite of the intuition most
people bring from cloud APIs.

**The box does not do concurrency. It queues.** Aggregate throughput was flat at roughly 24
tok/s whether one caller or eight were waiting, while wall time doubled at every step: 6.8s,
13.1s, 26.5s, 52.9s. Eight callers do not each get a slower stream, they get the same speed
in turn. Plan for one inference at a time. If you are running several programs against one
Tiiny, [OneLane](https://github.com/webdevtodayjason/onelane) makes them take turns properly
instead of guessing about each other.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TIINY_HOST` | `tiiny.local` | device address, host only. No scheme, no port |
| `TIINY_KEY` | *scraped* | device API key. See below |
| `TIINY_PORT` | `8800` | gateway port |

If `TIINY_KEY` is unset and you are on the Mac running TiinyOS, the suite digs the key out
of the app's own local storage and tries each candidate until the device accepts one. It is
a convenience for that one machine, not a security bypass. Set the variable anywhere else.

---

## Troubleshooting

**`No route to host` from Python but `curl` works.** macOS tracks Local Network permission
per binary, so Homebrew's Python and the system Python hold separate grants. Either allow
the one you are using in System Settings → Privacy & Security → Local Network, or run it
with `/usr/bin/python3`.

**`no model loaded`.** Load one in TiinyOS, or pass `--model` / `--all` and let the suite do
it.

**A model fails to load during a sweep.** It is recorded as a failure and the sweep carries
on. The box has 100 NPU units and some models want most of them, so make room first if you
are benchmarking the large ones.

---

## Credits

Built by [Jason Brashear](https://github.com/webdevtodayjason) at Titanium Computing.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tiiny-logo.svg">
  <img alt="Tiiny" src="assets/tiiny-logo-ink.svg" width="104">
</picture>

Measured on the [Tiiny Pocket](https://tiiny.ai/) by Tiiny AI.
