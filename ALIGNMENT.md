# Alignment Mode

Alignment mode aligns a known transcript against the audio to produce **word-level
timestamps** (forced alignment), rather than transcribing speech to text.

## Activating alignment mode

**Passing a text value to `--align` activates alignment mode.** The value is the
transcript you want aligned to the audio:

```bash
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "the quick brown fox"
```

If `--align` is omitted, the tool runs normal transcription instead — alignment
mode is entered **only** when a `--align <text>` value is supplied. There is no
default transcript; the text is required to enter the mode.

## Requirements

- A **ForcedAligner** model (e.g. `qwen3-aligner-0.6b`), not the regular ASR model.
  Download it with: `qwen-asr download qwen3-aligner-0.6b`
- An input source: `-i <file>` (WAV) or `--stdin`.

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--align <text>` | Transcript to align to the audio. **Supplying a value activates alignment mode.** | — (required to activate; no default) |
| `--align-language <lang>` | Language used to split the transcript into words | `English` |

`--align-language` is case-insensitive. Supported languages:

> Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese,
> Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish,
> Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino,
> Persian, Greek, Romanian, Hungarian, Macedonian

An unsupported value exits with an error listing the supported languages.

## Output

A JSON array of word entries with start/end timestamps **in milliseconds**, printed
to stdout:

```json
[
  {"text": "the", "start": 120, "end": 310},
  {"text": "quick", "start": 310, "end": 560},
  {"text": "brown", "start": 560, "end": 820},
  {"text": "fox", "start": 820, "end": 1100}
]
```

## Example

```bash
qwen-asr -d qwen3-aligner-0.6b -i audio.wav \
  --align "the quick brown fox" --align-language English
```
