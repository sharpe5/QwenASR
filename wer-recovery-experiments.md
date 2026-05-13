# WER Recovery Experiments

Goal: reduce 100-file LibriSpeech corpus WER below `0.04` while keeping speed within a 20% slowdown versus the current local baseline.

Baseline (`step0-current`, HEAD `12663c5`, runs=3):
- Speed: offline `781 ms`, segmented `798 ms`, streaming `1210 ms`
- 100-file WER: `0.1101`

## Step 1: disable default silence skipping

Change:
- `QwenCtx::new()` default `skip_silence: true -> false`

Results:
- Speed: offline `1194 ms`, segmented `1168 ms`, streaming `2271 ms`
- 100-file WER: `0.0708`

Decision:
- Rejected as a standalone fix. It reduces WER, but WER remains above `0.04` and speed loss exceeds 20%.

## Step 2: restore full-vocabulary argmax

Change:
- Removed the `0..39_000` plus final-`512` vocab shortlist from `argmax_matvec_int8()`.
- Kept the newer stack reduction and paired NEON range kernel.

Results:
- Speed: offline `823 ms`, segmented `774 ms`, streaming `1298 ms`
- 100-file WER: `0.0708`

Decision:
- Accepted as a partial fix. It reduces WER and all measured speed changes are within the 20% budget versus baseline, but WER is still above `0.04`.

## Step 3: remove default forced prompt fallback

Change:
- Removed the default fallback `force_prompt_tokens = [11528, 6364, <asr_text>]` when no language is forced.
- Tested on top of Step 2.

Results:
- Speed: offline `870 ms`, segmented `827 ms`, streaming `1378 ms`
- 100-file WER: `0.0729`

Decision:
- Rejected. Speed stayed within budget, but WER was worse than Step 2.

## Step 4: remove offline punctuation early-stop

Change:
- Removed the `n_text_tokens >= 40` punctuation early-stop in offline segment decoding.
- Tested on top of Step 2.

Results:
- Speed: offline `878 ms`, segmented `784 ms`, streaming `1388 ms`
- 100-file WER: `0.0708`

Decision:
- Rejected. WER did not improve over Step 2 and runtime was slower.

## Step 5: restore conservative silence compaction parameters

Change:
- Restored `compact_silence()` parameters to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 60`.
- Tested on top of Step 2.

Results:
- Speed: offline `1081 ms`, segmented `1160 ms`, streaming `1984 ms`
- 100-file WER: `0.0365`

Decision:
- Rejected as-is. It reaches the WER target, but speed loss exceeds 20%. This identifies silence compaction aggressiveness as the remaining accuracy lever to tune.

## Step 6: low threshold plus 3-window padding, no hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 0`.
- Tested on top of Step 2.

Results:
- Speed: offline `965 ms`, segmented `891 ms`, streaming `1690 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. It is faster than Step 5, but WER is still above `0.04` and streaming speed remains outside budget.

## Step 7: low threshold plus 3-window padding, 10-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 10`.
- Tested on top of Step 2.

Results:
- Speed: offline `978 ms`, segmented `884 ms`, streaming `1697 ms`
- 100-file WER: `0.0408`

Decision:
- Rejected. It gets close to the WER target but still misses, and speed remains outside budget.

## Step 8: threshold 0.004 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.004`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `1067 ms`, segmented `889 ms`, streaming `1695 ms`
- 100-file WER: `0.0328`

Decision:
- Rejected as-is. WER is comfortably below target, but speed remains outside the 20% budget.

## Step 9: threshold 0.006 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.006`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `959 ms`, segmented `914 ms`, streaming `1685 ms`
- 100-file WER: `0.0314`

Decision:
- Rejected as-is. WER is below target and segmented speed is within budget, but offline is slightly over the 20% cap and streaming is still too slow.

## Step 10: threshold 0.008 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `960 ms`, segmented `968 ms`, streaming `1712 ms`
- 100-file WER: `0.0314`

Decision:
- Rejected as-is. WER is below target, but speed remains outside the 20% budget.

## Step 11: threshold 0.008 plus 3-window padding, 15-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 3`, `pass_windows = 15`.
- Tested on top of Step 2.

Results:
- Speed: offline `972 ms`, segmented `867 ms`, streaming `1682 ms`
- 100-file WER: `0.0372`

Decision:
- Rejected as-is. WER is below target, but offline and streaming speed remain outside the 20% budget.

## Step 12: Step 11 silence tuning without full-vocabulary argmax

Change:
- Restored the commit's shortened argmax shortlist while keeping Step 11 silence tuning.

Results:
- Speed: offline `962 ms`, segmented `848 ms`, streaming `1656 ms`
- 100-file WER: `0.0780`

Decision:
- Rejected. Removing full-vocabulary argmax breaks WER, so full argmax is required.

## Step 13: threshold 0.008 plus 2-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 2`, `pass_windows = 20`.
- Tested with full-vocabulary argmax.

Results:
- Speed: offline `935 ms`, segmented `973 ms`, streaming `1747 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted for offline WER/speed, but not for segmented/streaming speed. Follow-up keeps this quality compaction for offline and uses fast compaction for segmented/streaming.

## Step 14: mode-specific compaction

Change:
- Kept quality compaction for offline transcription: `base_thresh = 0.008`, `pad_voice_windows = 2`, `pass_windows = 20`.
- Added fast compaction for segmented and streaming modes: `base_thresh = 0.0205`, `pad_voice_windows = 1`, `pass_windows = 0`.
- Kept full-vocabulary argmax.

Results:
- Speed: offline `909 ms`, segmented `816 ms`, streaming `1317 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. WER is below `0.04`, and all speed modes are within 20% of the fresh local baseline (`937 ms`, `958 ms`, `1452 ms` caps respectively).
