# What this benchmark does when the device says no

A benchmark is mostly a program for handling refusals. The numbers are the
easy part: you ask, you time it, you write it down. What decides whether the
file is worth anything a month later is what it does with the calls that did
not work.

This document is the rule the suite follows, written down after a day that
made the case for it better than an argument could.

## The rule

**Keep the body of every failed response, and act on what it says.**

Not the status code. Not the one-line error. The body, with the status and the
content type beside it. `capture()` does this, once per test per run, and the
record travels in the result file and through `public_view` like everything
else.

A test that can print a failure and does not call `capture()` fails the suite.
That is enforced in `tests/test_provenance.py`, not left to memory, because
the four chat tests went eight months without it and nobody noticed.

## The day that made the case

On 19 September 2026 a sweep of all 52 installed models ran for two hours and
forty-three minutes and came back with eleven measurements empty. Empty means
a `null` in the file, which reads exactly the same as a model nobody tried.

Reading the bodies took about twenty minutes and turned those eleven into:

| What the body said | What it turned out to be |
|---|---|
| `{"success":true,"audio_data":"UklGR..."}` | A working music model. The benchmark only understood a raw WAV or a job to poll, so it recorded a **failure for a model that had just produced audio**. |
| `Extra inputs are not permitted in request: duration` | Two music models that disagree about their own request shape. One requires `duration`, the other refuses it. |
| `Unsupported speaker: serena. Supported speakers: ['F1', ... 'M5']` | A speech model that works fine, once asked with a speaker it has. The gateway was choosing one it did not have. **The refusal contained the answer.** |
| `custom_voice is not supported by this model` | Two speech models that genuinely cannot be driven through this route. A real finding, and a different one from the line above. |
| `502 Bad Gateway` | Not broken at all. The device lists a model as running before its runtime accepts connections. Both models measured fine on a second attempt. |
| `Endpoint not found` / `Model ... does not support chat` | Four OCR models installable from the store and callable by nothing. |

Four of the eleven became real numbers. Four more became a diagnosis specific
enough to hand to the vendor. That is the whole return on keeping a string
somebody would otherwise have thrown away.

## What follows from it

**Read the refusal instead of keeping a table.** The device knows which fields
it takes and which speakers it has, and it says so when it says no. A table of
per-model quirks in this repo would be wrong within a week. `_rejected_field`,
`_required_fields` and `_offered_voices` all exist for this reason: they parse
what the box said and ask again properly.

**Never delete your own request to get past a validator.** SongGeneration
refuses `prompt` and then says a prompt is required. Two validators
disagreeing is not resolved by dropping fields until nothing is left, so
`NEVER_DROP` stops at `model` and `prompt`.

**A retry has to be bounded and it has to be honest.** Four field drops per
run, one voice attempt, one cold-runtime retry. Anything a retry changes about
the request goes in the record: which voice was used, which fields were
dropped.

**Do not guess where the box has not spoken.** An early version of the speech
fix walked a list of invented voice names. Story Lantern, which drives the
same route in production, already knew that naming a voice fails on those
models, and the sweep agreed. A refusal that enumerates its options is data.
A refusal that does not is a finding, and the file says so rather than
inventing a value.

**A stated reason beats a null.** `not_measured("...")` says which wall the
test hit. A bare `null` is indistinguishable from a test that never ran, and
that difference is the whole point of publishing the file.
