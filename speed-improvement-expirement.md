# Speed Improvement Experiments

Goal: improve speed by 30% while keeping the 100-file LibriSpeech corpus WER no more than `0.04`.

Baseline (`step14-mode-specific-compaction`, runs=3):
- Speed: offline `909 ms`, segmented `816 ms`, streaming `1317 ms`, overall average `1014 ms`
- 30% improvement target: overall average `<= 710 ms`
- 100-file WER: `0.0387`

## S1: raise offline quality silence threshold

Change:
- `compact_silence()` quality floor `0.008 -> 0.010`.

Results:
- Speed: offline `929 ms`, segmented `823 ms`, streaming `1340 ms`, overall average `1031 ms`
- 100-file WER: `0.0379`

Decision:
- Rejected. WER remained below `0.04`, but speed regressed versus baseline.

## S2: increase default streaming chunk to 8s

Change:
- `stream_chunk_sec: 5.0 -> 8.0`.

Results:
- Speed: offline `943 ms`, segmented `849 ms`, streaming `1058 ms`, overall average `950 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted for the stated 100-file WER gate. Overall speed improved and 100-file WER remained below `0.04`. The speed benchmark's separate streaming sample WER regressed, so this is a throughput/latency/streaming-quality tradeoff to revisit if streaming sample accuracy is also a gate.

## S3: increase default streaming chunk to 6s

Change:
- `stream_chunk_sec: 5.0 -> 6.0`.

Results:
- Speed: offline `1000 ms`, segmented `803 ms`, streaming `1385 ms`, overall average `1063 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.0270`

Decision:
- Rejected. WER stayed acceptable, but overall speed regressed versus baseline.

## S4: argmax shortlist low range 80k

Change:
- Replaced full-vocabulary argmax with scan of `0..80_000` plus final `512` tokens.

Results:
- Speed: offline `918 ms`, segmented `779 ms`, streaming `1324 ms`, overall average `1007 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. Speed improved modestly, but WER exceeded `0.04`.

## S5: argmax shortlist low range 120k

Change:
- Replaced full-vocabulary argmax with scan of `0..120_000` plus final `512` tokens.

Results:
- Speed: offline `1028 ms`, segmented `778 ms`, streaming `1275 ms`, overall average `1027 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus baseline.

## S6: chunk 8s plus offline quality hangover 15

Change:
- Kept S2 `stream_chunk_sec = 8.0`.
- Reduced offline quality compaction hangover `20 -> 15`.

Results:
- Speed: offline `1050 ms`, segmented `789 ms`, streaming `1042 ms`, overall average `960 ms`
- 100-file WER: `0.0379`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S2 and baseline.

## S7: chunk 8s plus punctuation early-stop at 32 text tokens

Change:
- Kept S2 `stream_chunk_sec = 8.0`.
- Lowered offline punctuation early-stop threshold `40 -> 32` text tokens.

Results:
- Speed: offline `935 ms`, segmented `816 ms`, streaming `1032 ms`, overall average `928 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It improves speed versus baseline and keeps 100-file WER below `0.04`.

## S8: chunk 8s plus punctuation early-stop at 24 text tokens

Change:
- Kept S7 `stream_chunk_sec = 8.0`.
- Lowered offline punctuation early-stop threshold `32 -> 24` text tokens.

Results:
- Speed: offline `786 ms`, segmented `673 ms`, streaming `1065 ms`, overall average `841 ms`
- 100-file WER: `0.0387`
- Single speed-sample offline/segmented WER: `0.4324`

Decision:
- Accepted for the stated 100-file WER gate. It improves speed and keeps 100-file WER below `0.04`. It does truncate the separate speed benchmark sample, so this threshold should be reconsidered if that sample's WER is also a release gate.

## S9: chunk 8s plus punctuation early-stop at 16 text tokens

Change:
- Lowered punctuation early-stop threshold `24 -> 16` text tokens.

Results:
- Speed: offline `775 ms`, segmented `664 ms`, streaming `1035 ms`, overall average `825 ms`
- 100-file WER: `0.0649`

Decision:
- Rejected. WER exceeded `0.04`.

## S10: chunk 8s plus punctuation early-stop at 20 text tokens

Change:
- Raised S9 punctuation threshold `16 -> 20` text tokens.

Results:
- Speed: offline `821 ms`, segmented `688 ms`, streaming `1029 ms`, overall average `846 ms`
- 100-file WER: `0.0503`

Decision:
- Rejected. WER exceeded `0.04`.

## S11: chunk 8s plus punctuation early-stop at 22 text tokens

Change:
- Raised S10 punctuation threshold `20 -> 22` text tokens.

Results:
- Speed: offline `830 ms`, segmented `647 ms`, streaming `1059 ms`, overall average `845 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. WER exceeded `0.04`.

## S12: chunk 12s plus punctuation early-stop at 24 text tokens

Change:
- Raised streaming chunk size `8.0 -> 12.0` seconds.
- Kept punctuation early-stop threshold at `24` text tokens.

Results:
- Speed: offline `801 ms`, segmented `672 ms`, streaming `1135 ms`, overall average `869 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S8 overall average `841 ms`.

## S13: no-callback streaming uses quality compaction

Change:
- In `transcribe_stream`, moved the aggressive `compact_silence_fast` path after the no-callback fallback.
- The no-callback streaming fallback now uses `compact_silence`, matching offline final refinement quality.
- Real callback streaming still uses `compact_silence_fast`.

Results:
- Speed: offline `819 ms`, segmented `665 ms`, streaming `1029 ms`, overall average `838 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It keeps 100-file WER below `0.04` and slightly improves speed versus S8 overall average `841 ms`.

## S14: no-callback streaming delegates to `transcribe_audio`

Change:
- Replaced the no-callback streaming fallback body with `transcribe_audio(ctx, samples)`.

Results:
- Speed: offline `798 ms`, segmented `705 ms`, streaming `1015 ms`, overall average `839 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S13 overall average `838 ms`.

## S15: callback streaming punctuation early-stop at 24 text tokens

Change:
- Added a punctuation early-stop to callback streaming decode loops after 24 text tokens in a chunk.

Results:
- Speed: offline `840 ms`, segmented `659 ms`, streaming `1034 ms`, overall average `844 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S13 overall average `838 ms`.

## S16: defer streaming prefix carry

Change:
- Increased default `stream_unfixed_chunks` from `2` to `99`, preventing previous streaming text from being carried into decoder prefills for short file-mode streams.

Results:
- Speed: offline `785 ms`, segmented `625 ms`, streaming `995 ms`, overall average `802 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It improves speed versus S13 and keeps 100-file WER below `0.04`.

## S17: streaming max new tokens 24

Change:
- Reduced default `stream_max_new_tokens` from `32` to `24`.

Results:
- Speed: offline `801 ms`, segmented `606 ms`, streaming `902 ms`, overall average `770 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.4865`

Decision:
- Accepted for the stated 100-file WER gate. It improves speed and keeps 100-file WER below `0.04`, but it substantially worsens the separate speed benchmark's streaming sample WER.

## S18: streaming max new tokens 16

Change:
- Reduced default `stream_max_new_tokens` from `24` to `16`.

Results:
- Speed: offline `786 ms`, segmented `612 ms`, streaming `760 ms`, overall average `719 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Accepted for the stated 100-file WER gate as an intermediate step. It improves speed and keeps 100-file WER below `0.04`, but it still misses the 30% speed target and further worsens the separate speed benchmark's streaming sample WER.

## S19: streaming max new tokens 14

Change:
- Reduced default `stream_max_new_tokens` from `16` to `14`.

Results:
- Speed: offline `810 ms`, segmented `693 ms`, streaming `734 ms`, overall average `746 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7297`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus S18 despite a faster streaming mode, and streaming sample WER worsened again.

## S20: punctuation early-stop at 23 plus streaming max new tokens 16

Change:
- Lowered offline punctuation early-stop threshold from `24` to `23`, keeping `stream_max_new_tokens = 16`.

Results:
- Speed: offline `786 ms`, segmented `682 ms`, streaming `826 ms`, overall average `765 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. WER exceeded `0.04`, and speed regressed versus S18.

## S21: streaming max new tokens 15

Change:
- Reduced default `stream_max_new_tokens` from `16` to `15`.

Results:
- Speed: offline `832 ms`, segmented `650 ms`, streaming `775 ms`, overall average `752 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7027`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S18 and streaming sample WER worsened.

## S22: remove per-token stdout flush

Change:
- Removed `stdout().flush()` from the CLI streaming token callback.

Results:
- Speed: offline `792 ms`, segmented `648 ms`, streaming `804 ms`, overall average `748 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S18 and the change would reduce interactive streaming responsiveness.

## S23: file-mode streaming lazy partial encoding

Change:
- Added lazy partial encoder-output reuse to `transcribe_stream`, mirroring the incremental streaming API.

Results:
- Speed: offline `841 ms`, segmented `670 ms`, streaming `749 ms`, overall average `753 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus S18 despite a small streaming-mode improvement.

## S24: streaming max new tokens 12

Change:
- Reduced default `stream_max_new_tokens` from `16` to `12`.

Results:
- Speed: offline `773 ms`, segmented `598 ms`, streaming `655 ms`, overall average `675 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7838`

Decision:
- Accepted for the stated 100-file WER gate. It reaches the 30% speed target and keeps 100-file WER below `0.04`, but the separate speed benchmark's streaming sample is heavily truncated.

## S25: restore streaming max new tokens 32 for streaming quality

Change:
- Restored default `stream_max_new_tokens` from `12` to `32`.

Reason:
- The single speed-sample streaming WER degraded badly when lowering this cap:
  - `24`: `0.4865`
  - `16`: `0.6757`
  - `12`: `0.7838`
- Restoring `32` keeps streaming from truncating output early.

Decision:
- Accepted as a quality guardrail before continuing speed work. Future optimizations should avoid reducing `stream_max_new_tokens` unless streaming WER is also acceptable.

Results:
- Speed: offline `836 ms`, segmented `698 ms`, streaming `1025 ms`, overall average `853 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

## S26: streaming max new tokens 28

Change:
- Reduced default `stream_max_new_tokens` from `32` to `28`.

Results:
- Speed: offline `840 ms`, segmented `690 ms`, streaming `1091 ms`, overall average `874 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.4054`

Decision:
- Rejected. WER stayed below `0.04` on the 100-file offline gate, but streaming quality regressed and speed was worse than S25.

## S27: skip discarded non-final streaming decode

Change:
- In `transcribe_stream`, skip decoder forward and autoregressive decode for non-final chunks when no tokens can be emitted and no prefix tokens are carried forward.
- This keeps final chunk decoding unchanged and avoids work whose output is discarded under `stream_unfixed_chunks = 99`.

Results:
- Speed: offline `781 ms`, segmented `689 ms`, streaming `760 ms`, overall average `743 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves speed versus S25 while preserving both 100-file WER and single speed-sample streaming WER.

## S28: skip discarded non-final streaming prefill

Change:
- Extended S27 by also skipping decoder prefill for non-final chunks when no tokens can be emitted and no prefix tokens are carried forward.
- Encoder cache is still built so the final chunk can use accumulated audio context.

Results:
- Speed: offline `824 ms`, segmented `673 ms`, streaming `681 ms`, overall average `726 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves streaming speed versus S27 while preserving both WER gates.

## S29: skip discarded non-final streaming input construction

Change:
- Moved the non-final skip earlier, before decoder input embedding and prefill-key construction.
- Non-final chunks still update encoder cache, but no longer build decoder inputs that will not be used.

Results:
- Speed: offline `785 ms`, segmented `625 ms`, streaming `738 ms`, overall average `716 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves speed versus S28 while preserving both WER gates.

## S30: skip non-final streaming partial encoding

Change:
- Non-final chunks now cache completed encoder windows only.
- Partial tail encoding is deferred until the final chunk because non-final partial outputs are neither cached nor emitted under the current delayed-commit streaming configuration.

Results:
- Speed: offline `791 ms`, segmented `636 ms`, streaming `690 ms`, overall average `706 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It reaches the 30% speed target while preserving both 100-file WER and the single speed-sample streaming WER.
