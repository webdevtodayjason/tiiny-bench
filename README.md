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
./tiiny-bench --where                   # find the box, say how it was reached
./tiiny-bench --selfcheck               # prove the install works, ten seconds
./tiiny-bench --catalog                 # what is installed
./tiiny-bench --label first-run         # benchmark whatever is loaded
./tiiny-bench --report                  # build report.html
```

No address to set. It finds the box itself. Python standard library only, nothing to
install.

---

## What it measures

| Test | The question it answers |
|---|---|
| **Prefill scaling** | What does a long prompt cost to read? Four prompt lengths from ~70 to ~6000 tokens, each asking for a 4-token answer so decode barely registers. |
| **Sustained generation** | Does throughput hold? One unbroken 1500-token generation, with NPU utilisation sampled on a thread *while it runs*. |
| **Concurrency** | 1, 2, 4 and 8 identical requests fired at once. Aggregate throughput against per-stream throughput. |
| **Reasoning cost** | The same question with thinking off and then on. The ratio is wall time, because that is what a person waits. |

Those four are for the models that hold a conversation, and they are a choice: pick any
combination. Every other class the box ships has one test and runs it, so there is nothing
to pick.

| Class | Test | The figure |
|---|---|---|
| **Text-to-Image** | Three 512x512 plates at 8 steps. | seconds per plate |
| **Text-to-Speech** | Three passages spoken, duration read off the WAV header. | times faster than real time |
| **Text Embedding** | Batches of 1, 8 and 32. | embeddings per second |
| **ASR** | Clips of 2, 5 and 10 seconds transcribed. | times faster than real time |
| **Image-to-Text** | A generated page of digits read three times, through the OCR gateway or through chat completions, whichever answers. The reply says which model read it, and that is recorded rather than assumed. | seconds per page |
| **Music Generation** | 8 and 16 seconds asked for, blocking or polled. | seconds of audio per second of wall clock |
| **Text Reranking** | 4, 16 and 64 passages scored against one query. | query-document pairs per second |

The audio clip and the page of digits are generated in code, not committed: a fixture you
cannot diff is one you cannot trust when a number moves.

**What it does not measure: quality.** Nothing here says a model is good, only how fast it
is. A fast wrong answer is still wrong. The two exceptions are small and free: page reading
says whether the digits came back, and reranking says whether the passage that answers the
query ranked first, because a reranker that is fast and wrong is worth knowing about.

---

## Chat

`--serve` also opens a **Chat** page, because the fastest way to understand what a number
means is to sit behind it. It is the same box asked a question instead of a benchmark, and
under every reply is what that reply cost:

```
ttft 883 ms   decode 23.0 tok/s   total 4.21 s   in 34   out 74
on My Tiiny · Ornith-1.0-35B   finish stop
```

Every one of those figures is the device's own, read off the gateway's `timings` and `usage`
through the same `derive_stats` the benchmark uses, plus this machine's clock for the two
wall-time figures it alone can see. A stream that fails part way reports nothing rather than
zeroes, because "out 0" beside a turn that really streamed four hundred tokens is a lie the
error above it already contradicts.

Left of the conversation is a card for the selected model, and the last twenty conversations,
which are kept in your browser and never leave it. Right of it is every model loaded on the
box with its NPU units and an Unload button, the budget bar, and a Load panel that refuses in
a sentence when a model is not installed, cannot chat, or would not fit:

> Qwen3.6-35B-A3B-turbo asks for 55 NPU units and only 42 are free on My Tiiny. The device
> would accept the load and roll it back a moment later without saying so, so it is refused
> here.

That refusal is the point of the panel. The device does not say no to a load that does not
fit; it says yes, shows the model as loading, and drops it a moment later with no error
anywhere. The Models page loads through the same guard, so it refuses on the same grounds.

Replies render as markdown with highlighted code blocks and a Copy button, and a reasoning
model's chain of thought sits in a collapsible block behind the Thinking toggle. Temperature,
max tokens and a system prompt are controls on the page. Only
`chat_template_kwargs.enable_thinking` turns reasoning on or off on this gateway: the top
level `enable_thinking` its own OpenAPI document declares is ignored by the runtime, measured.

The page is still one self-contained HTML file. No CDN, no external script, no webfont, and
nothing a model writes reaches the page as markup.

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

## Finding your Tiiny

Firmware 1.0 changed how a box is addressed, twice over. The AI gateway no longer answers
on port 8800 from another machine: it binds the container bridge only, and every service now
arrives on port 80 where a router picks one out of the `Host` header. And a box's LAN address
is a DHCP lease, so it moves.

So there is nothing to configure. The suite works out the address, in this order, and always
prints which step answered:

| Step | What it is |
|---|---|
| `TIINY_BASE` | what the farm CLI puts in the environment. `TIINY_HOST` still works |
| `~/.tiinyapps/device.json` | what `farm device` writes: `{"base": ..., "key": ...}` |
| `--host 192.168.1.50` | an address you type. The two above win over it, and say so |
| the saved config | whatever the last successful run found. `--rescan` ignores it |
| a scan | every USB `/30` peer, then this machine's own `/24`, on `:39218` |

The scan is the only step that identifies a box rather than just finding an open port,
because `:39218/device.json` carries the serial. That is what makes the rest work:

- **A box on Wi-Fi and USB at once is one box, not two.** The two addresses are deduped by
  serial, and USB wins, because a `/30` handed out by the cable cannot move and a DHCP lease
  can.
- **More than one box is a question, not a guess.** If two serials answer, the suite lists
  them and stops. Pick one with `--serial TNYM...` or `--host`.

Then the transport: the service's own port is tried first, and **only** a refused connection
moves it to the port 80 vhost for the rest of the run. A timeout does not, because a busy box
is not old firmware. Which address, which plane and which transport produced a set of numbers
all go in the result file, next to both firmware versions, because a run over USB and a run
over the port 80 vhost are not the same measurement.

```bash
./tiiny-bench --where
#   address    192.168.100.94
#   found by   scan
#   plane      lan
#   serial     TNYM26072400300011Q
#   gateway    port 80, Host: p8800.api.tiiny   (vhost)
#   firmware   TiinyOS 0.1.34  service 0.1.30
```

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TIINY_BASE` | *found* | device address or base URL. A port in it overrides the gateway's |
| `TIINY_HOST` | *found* | the old name for `TIINY_BASE`. Still honoured |
| `TIINY_KEY` | *scraped* | device API key. See below |
| `TIINY_PORT` | `8800` | the gateway's own port, if you have moved it |

The key follows the same idea: `TIINY_KEY`, then `~/.tiinyapps/device.json`, then the saved
config, then, on the Mac running TiinyOS, the app's own local storage. That last one reads
both the `.ldb` and `.log` halves of the app's LevelDB and tries the most frequently seen
candidate first, because LevelDB writes the newest value to the log and only compacts it
later, so looking at one half finds a key that is either stale or missing. It is a
convenience for that one machine, not a security bypass. Set the variable anywhere else.

---

## Troubleshooting

**`No route to host` from Python but `curl` works.** macOS tracks Local Network permission
per binary, so Homebrew's Python and the system Python hold separate grants. Either allow
the one you are using in System Settings → Privacy & Security → Local Network, or run it
with `/usr/bin/python3`.

**`no model loaded`.** Load one in TiinyOS, or pass `--model` / `--all` and let the suite do
it.

**`No Tiiny found`.** Run `./tiiny-bench --where` to see what was tried. The scan only sweeps
the `/24` this machine sits in, so a box on another subnet needs `--host`. A box that is only
on USB needs the cable actually attached: check for a `172.17.x.x` address in `ifconfig`.

**`More than one Tiiny answered`.** That is not an error, it is the suite refusing to pick for
you. Take the serial it printed and pass `--serial`, or pass `--host`.

**Numbers that do not match an older run.** Check the `connection` block in both result files.
The same box measured over USB and over the LAN vhost are two different measurements, and the
result file records which one each was.

**Every inference fails but the box is plainly there.** The address and the transport are
cached, and a cached one can go stale: the gateway moved from port 8800 to port 80 in TiinyOS
1.0.0, and a saved 8800 keeps being re-chosen. Detect again does not help, because it prefers
what it already has. Restart the app, which makes it negotiate the transport from scratch.
`./tiiny-bench --where` prints the port and vhost it settled on, and a result file records the
same thing in its `connection` block.

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
