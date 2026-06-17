//! Audio encoder: Conv2D stem + windowed transformer + projection cascade.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

/// A dense weight matrix held either as f32 (dequantized at load — the default) or
/// BF16-resident (a raw pointer into the mmap'd safetensors, widened to f32 per call).
/// BF16-resident roughly halves weight RAM; see `--weights bf16`. The matmul math is
/// identical either way (BF16→f32 is an exact widening), only WHEN the widen happens
/// differs. The pointer is valid for the program lifetime (the `MultiSafetensors` mmap
/// is kept alive by `QwenCtx._safetensors`).
pub enum Wt {
    F32(Vec<f32>),
    Bf16(*const u16),
}

unsafe impl Send for Wt {}
unsafe impl Sync for Wt {}

impl Wt {
    /// Stable identity of this weight matrix (for the ANE model cache).
    #[cfg(all(target_os = "macos", feature = "mac-ane"))]
    #[inline]
    fn ptr_key(&self) -> usize {
        match self {
            Wt::F32(w) => w.as_ptr() as usize,
            Wt::Bf16(p) => *p as usize,
        }
    }

    /// Row-major `[out_dim, in_dim]` f32 weights (widening BF16 if needed).
    /// Only called on an ANE cache miss.
    #[cfg(all(target_os = "macos", feature = "mac-ane"))]
    fn to_f32_weights(&self, in_dim: usize, out_dim: usize) -> Vec<f32> {
        match self {
            Wt::F32(w) => w.clone(),
            Wt::Bf16(p) => {
                let n = in_dim * out_dim;
                let src = unsafe { std::slice::from_raw_parts(*p, n) };
                let mut v = vec![0.0f32; n];
                kernels::bf16_to_f32_buf(&mut v, src);
                v
            }
        }
    }

    #[inline]
    fn linear(&self, y: &mut [f32], x: &[f32], b: Option<&[f32]>, seq: usize, in_dim: usize, out_dim: usize) {
        #[cfg(all(target_os = "macos", feature = "mac-ane"))]
        {
            if let Some(ym) = crate::mac_ane::try_linear(
                self.ptr_key(), x, in_dim, out_dim, seq,
                || self.to_f32_weights(in_dim, out_dim),
            ) {
                match b {
                    Some(b) => {
                        for s in 0..seq {
                            for o in 0..out_dim {
                                y[s * out_dim + o] = ym[s * out_dim + o] + b[o];
                            }
                        }
                    }
                    None => y[..seq * out_dim].copy_from_slice(&ym),
                }
                return;
            }
        }
        match self {
            Wt::F32(w) => kernels::linear(y, x, w, b, seq, in_dim, out_dim),
            Wt::Bf16(p) => kernels::linear_bf16(y, x, *p, b, seq, in_dim, out_dim),
        }
    }
    #[inline]
    fn linear_nobias(&self, y: &mut [f32], x: &[f32], seq: usize, in_dim: usize, out_dim: usize) {
        #[cfg(all(target_os = "macos", feature = "mac-ane"))]
        {
            if let Some(ym) = crate::mac_ane::try_linear(
                self.ptr_key(), x, in_dim, out_dim, seq,
                || self.to_f32_weights(in_dim, out_dim),
            ) {
                y[..seq * out_dim].copy_from_slice(&ym);
                return;
            }
        }
        match self {
            Wt::F32(w) => kernels::linear_nobias(y, x, w, seq, in_dim, out_dim),
            Wt::Bf16(p) => kernels::linear_nobias_bf16(y, x, *p, seq, in_dim, out_dim),
        }
    }
    #[inline]
    fn linear_accumulate(&self, y: &mut [f32], x: &[f32], b: Option<&[f32]>, seq: usize, in_dim: usize, out_dim: usize) {
        #[cfg(all(target_os = "macos", feature = "mac-ane"))]
        {
            if let Some(ym) = crate::mac_ane::try_linear(
                self.ptr_key(), x, in_dim, out_dim, seq,
                || self.to_f32_weights(in_dim, out_dim),
            ) {
                match b {
                    Some(b) => {
                        for s in 0..seq {
                            for o in 0..out_dim {
                                y[s * out_dim + o] += ym[s * out_dim + o] + b[o];
                            }
                        }
                    }
                    None => {
                        for i in 0..seq * out_dim {
                            y[i] += ym[i];
                        }
                    }
                }
                return;
            }
        }
        match self {
            Wt::F32(w) => kernels::linear_accumulate(y, x, w, b, seq, in_dim, out_dim),
            Wt::Bf16(p) => kernels::linear_accumulate_bf16(y, x, *p, b, seq, in_dim, out_dim),
        }
    }
}

pub struct EncLayer {
    pub wq_weight: Wt,
    pub wq_bias: Vec<f32>,
    pub wk_weight: Wt,
    pub wk_bias: Vec<f32>,
    pub wv_weight: Wt,
    pub wv_bias: Vec<f32>,
    pub wo_weight: Wt,
    pub wo_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub fc1_weight: Wt,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Wt,
    pub fc2_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
}

unsafe impl Send for EncLayer {}
unsafe impl Sync for EncLayer {}

pub struct EncoderBuffers {
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub chunk_mel: Vec<f32>,
    pub c1: Vec<f32>,
    pub c2: Vec<f32>,
    pub c3: Vec<f32>,
    pub reshaped: Vec<f32>,
    pub pe: Vec<f32>,
    pub conv_cols: Vec<f32>,
    pub window_starts: Vec<i32>,
    pub cap_tokens: usize,
}

impl Default for EncoderBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderBuffers {
    pub fn new() -> Self {
        EncoderBuffers {
            x: Vec::new(),
            x_norm: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            proj_out: Vec::new(),
            ffn_mid: Vec::new(),
            ffn_out: Vec::new(),
            chunk_mel: Vec::new(),
            c1: Vec::new(),
            c2: Vec::new(),
            c3: Vec::new(),
            reshaped: Vec::new(),
            pe: Vec::new(),
            conv_cols: Vec::new(),
            window_starts: Vec::new(),
            cap_tokens: 0,
        }
    }

    pub fn ensure(&mut self, total_tokens: usize, d_model: usize, ffn_dim: usize) {
        if total_tokens <= self.cap_tokens {
            return;
        }
        let mut new_cap = if self.cap_tokens > 0 {
            self.cap_tokens
        } else {
            256
        };
        while new_cap < total_tokens {
            new_cap *= 2;
        }
        self.x.resize(new_cap * d_model, 0.0);
        self.x_norm.resize(new_cap * d_model, 0.0);
        self.q.resize(new_cap * d_model, 0.0);
        self.k.resize(new_cap * d_model, 0.0);
        self.v.resize(new_cap * d_model, 0.0);
        self.attn_out.resize(new_cap * d_model, 0.0);
        self.proj_out.resize(new_cap * d_model, 0.0);
        self.ffn_mid.resize(new_cap * ffn_dim, 0.0);
        self.ffn_out.resize(new_cap * d_model, 0.0);
        self.cap_tokens = new_cap;
    }

    pub fn ensure_stem(&mut self, chunk_w: usize, d_model: usize) {
        let h1 = (128 + 2 - 3) / 2 + 1;
        let w1 = (chunk_w + 2 - 3) / 2 + 1;
        let h2 = (h1 + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let h3 = (h2 + 2 - 3) / 2 + 1;
        let w3 = (w2 + 2 - 3) / 2 + 1;
        let conv_proj_dim = CONV_HIDDEN * h3;

        self.chunk_mel.resize(128 * chunk_w, 0.0);
        self.c1.resize(CONV_HIDDEN * h1 * w1, 0.0);
        self.c2.resize(CONV_HIDDEN * h2 * w2, 0.0);
        self.c3.resize(CONV_HIDDEN * h3 * w3, 0.0);
        self.reshaped.resize(w3 * conv_proj_dim, 0.0);
        self.pe.resize(w3 * d_model, 0.0);
    }
}

pub struct Encoder {
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv3_weight: Vec<f32>,
    pub conv3_bias: Vec<f32>,
    pub conv_out_weight: Wt,
    pub layers: Vec<EncLayer>,
    pub ln_post_weight: Vec<f32>,
    pub ln_post_bias: Vec<f32>,
    pub proj1_weight: Wt,
    pub proj1_bias: Vec<f32>,
    pub proj2_weight: Wt,
    pub proj2_bias: Vec<f32>,
}

unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

const ENC_PREFIX: &str = "thinker.audio_tower.";

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("encoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_as_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let (si, t) = ms.find(name).or_else(|| {
        eprintln!("encoder: weight not found: {}", name);
        None
    })?;

    let bf16_ptr = ms.shards[si].get_bf16_direct(t)?;
    let n = t.numel();
    let mut f32_data = vec![0.0f32; n];
    let src = unsafe { std::slice::from_raw_parts(bf16_ptr, n) };
    for i in 0..n {
        f32_data[i] = f32::from_bits((src[i] as u32) << 16);
    }
    Some(f32_data)
}

/// Load a dense weight as f32 (dequantized) or BF16-resident (raw mmap pointer),
/// per the `weights_bf16` switch. The matmul result is identical; only RAM differs.
fn load_wt(ms: &MultiSafetensors, name: &str, weights_bf16: bool) -> Option<Wt> {
    if weights_bf16 {
        let (si, t) = ms.find(name).or_else(|| {
            eprintln!("encoder: weight not found: {}", name);
            None
        })?;
        let ptr = ms.shards[si].get_bf16_direct(t)?;
        Some(Wt::Bf16(ptr))
    } else {
        Some(Wt::F32(load_bf16_as_f32(ms, name)?))
    }
}

impl Encoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig, weights_bf16: bool) -> Option<Self> {
        let p = ENC_PREFIX;

        let conv1_weight = load_f32(ms, &format!("{}conv2d1.weight", p))?;
        let conv1_bias = load_f32(ms, &format!("{}conv2d1.bias", p))?;
        let conv2_weight = load_f32(ms, &format!("{}conv2d2.weight", p))?;
        let conv2_bias = load_f32(ms, &format!("{}conv2d2.bias", p))?;
        let conv3_weight = load_f32(ms, &format!("{}conv2d3.weight", p))?;
        let conv3_bias = load_f32(ms, &format!("{}conv2d3.bias", p))?;
        let conv_out_weight = load_wt(ms, &format!("{}conv_out.weight", p), weights_bf16)?;

        let mut layers = Vec::new();
        for i in 0..cfg.enc_layers {
            let lp = format!("{}layers.{}", p, i);

            let layer = EncLayer {
                wq_weight: load_wt(ms, &format!("{}.self_attn.q_proj.weight", lp), weights_bf16)?,
                wq_bias: load_f32(ms, &format!("{}.self_attn.q_proj.bias", lp))?,
                wk_weight: load_wt(ms, &format!("{}.self_attn.k_proj.weight", lp), weights_bf16)?,
                wk_bias: load_f32(ms, &format!("{}.self_attn.k_proj.bias", lp))?,
                wv_weight: load_wt(ms, &format!("{}.self_attn.v_proj.weight", lp), weights_bf16)?,
                wv_bias: load_f32(ms, &format!("{}.self_attn.v_proj.bias", lp))?,
                wo_weight: load_wt(ms, &format!("{}.self_attn.out_proj.weight", lp), weights_bf16)?,
                wo_bias: load_f32(ms, &format!("{}.self_attn.out_proj.bias", lp))?,
                attn_norm_weight: load_f32(ms, &format!("{}.self_attn_layer_norm.weight", lp))?,
                attn_norm_bias: load_f32(ms, &format!("{}.self_attn_layer_norm.bias", lp))?,
                fc1_weight: load_wt(ms, &format!("{}.fc1.weight", lp), weights_bf16)?,
                fc1_bias: load_f32(ms, &format!("{}.fc1.bias", lp))?,
                fc2_weight: load_wt(ms, &format!("{}.fc2.weight", lp), weights_bf16)?,
                fc2_bias: load_f32(ms, &format!("{}.fc2.bias", lp))?,
                ffn_norm_weight: load_f32(ms, &format!("{}.final_layer_norm.weight", lp))?,
                ffn_norm_bias: load_f32(ms, &format!("{}.final_layer_norm.bias", lp))?,
            };
            layers.push(layer);
        }

        let ln_post_weight = load_f32(ms, &format!("{}ln_post.weight", p))?;
        let ln_post_bias = load_f32(ms, &format!("{}ln_post.bias", p))?;
        let proj1_weight = load_wt(ms, &format!("{}proj1.weight", p), weights_bf16)?;
        let proj1_bias = load_f32(ms, &format!("{}proj1.bias", p))?;
        let proj2_weight = load_wt(ms, &format!("{}proj2.weight", p), weights_bf16)?;
        let proj2_bias = load_f32(ms, &format!("{}proj2.bias", p))?;

        Some(Encoder {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            conv3_weight,
            conv3_bias,
            conv_out_weight,
            layers,
            ln_post_weight,
            ln_post_bias,
            proj1_weight,
            proj1_bias,
            proj2_weight,
            proj2_bias,
        })
    }

    /// Conv2D stem for ONE segment: mel `[128, mel_frames]` → the post-stem
    /// sequence `[total_tokens, d_model]` (Conv2D ×3 → reshape → conv_out
    /// projection → per-chunk sinusoidal PE), returned as an owned buffer plus
    /// its `total_tokens`. `bufs` supplies only the transient conv scratch
    /// (chunk_mel / c1..c3 / reshaped / pe / conv_cols); the returned sequence is
    /// independent of `bufs.x`, so a caller can run this for several segments and
    /// concatenate the results (see [`forward_batch`]).
    fn conv_stem_forward(
        &self,
        cfg: &QwenConfig,
        mel: &[f32],
        mel_frames: usize,
        bufs: &mut EncoderBuffers,
    ) -> (Vec<f32>, usize) {
        let d_model = cfg.enc_d_model;
        let chunk_size = cfg.enc_chunk_size;
        let n_chunks = mel_frames.div_ceil(chunk_size);

        let mut total_tokens = 0;
        let mut chunk_sizes = Vec::new();
        for c in 0..n_chunks {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(mel_frames);
            let chunk_w = end - start;
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let w3 = (w2 + 2 - 3) / 2 + 1;
            total_tokens += w3;
            chunk_sizes.push((start, end, w3));
        }

        let mut out = vec![0.0f32; total_tokens * d_model];
        let mut token_offset = 0;
        for &(start, end, w3) in &chunk_sizes {
            let chunk_w = end - start;
            bufs.ensure_stem(chunk_w, d_model);

            let chunk_mel = &mut bufs.chunk_mel[..128 * chunk_w];
            for m in 0..128 {
                chunk_mel[m * chunk_w..(m + 1) * chunk_w]
                    .copy_from_slice(&mel[m * mel_frames + start..m * mel_frames + end]);
            }

            let h1 = (128 + 2 - 3) / 2 + 1; // 64
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let c1 = &mut bufs.c1[..CONV_HIDDEN * h1 * w1];
            kernels::conv2d_with_cols(
                c1, chunk_mel, &self.conv1_weight, Some(&self.conv1_bias),
                &mut bufs.conv_cols, 1, CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1,
            );
            kernels::gelu(c1, CONV_HIDDEN * h1 * w1);

            let h2 = (h1 + 2 - 3) / 2 + 1; // 32
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let c2 = &mut bufs.c2[..CONV_HIDDEN * h2 * w2];
            kernels::conv2d_with_cols(
                c2, c1, &self.conv2_weight, Some(&self.conv2_bias),
                &mut bufs.conv_cols, CONV_HIDDEN, CONV_HIDDEN, h1, w1, 3, 3, 2, 1,
            );
            kernels::gelu(c2, CONV_HIDDEN * h2 * w2);

            let h3 = (h2 + 2 - 3) / 2 + 1; // 16
            let _w3_calc = (w2 + 2 - 3) / 2 + 1;
            debug_assert_eq!(_w3_calc, w3);
            let c3 = &mut bufs.c3[..CONV_HIDDEN * h3 * w3];
            kernels::conv2d_with_cols(
                c3, c2, &self.conv3_weight, Some(&self.conv3_bias),
                &mut bufs.conv_cols, CONV_HIDDEN, CONV_HIDDEN, h2, w2, 3, 3, 2, 1,
            );
            kernels::gelu(c3, CONV_HIDDEN * h3 * w3);

            // Reshape [480, h3, w3] -> [w3, 480*h3]
            let conv_proj_dim = CONV_HIDDEN * h3;
            let reshaped = &mut bufs.reshaped[..w3 * conv_proj_dim];
            for ch in 0..CONV_HIDDEN {
                for f in 0..h3 {
                    let src_off = ch * h3 * w3 + f * w3;
                    let dst_col = ch * h3 + f;
                    for t in 0..w3 {
                        reshaped[t * conv_proj_dim + dst_col] = c3[src_off + t];
                    }
                }
            }

            // Project [w3, 7680] -> [w3, d_model] (+ sinusoidal PE) into `out`.
            let projected = &mut out[token_offset * d_model..(token_offset + w3) * d_model];
            self.conv_out_weight.linear_nobias(projected, reshaped, w3, conv_proj_dim, d_model);
            let pe = &mut bufs.pe[..w3 * d_model];
            kernels::sinusoidal_pe(pe, w3, d_model);
            kernels::add_inplace(projected, pe, w3 * d_model);

            token_offset += w3;
        }
        (out, total_tokens)
    }

    /// Run encoder forward pass on mel spectrogram.
    /// mel: [128, mel_frames], returns [total_tokens, output_dim].
    pub fn forward(
        &self,
        cfg: &QwenConfig,
        mel: &[f32],
        mel_frames: usize,
        enc_bufs: Option<&mut EncoderBuffers>,
    ) -> Option<(Vec<f32>, usize)> {
        let d_model = cfg.enc_d_model;
        let n_heads = cfg.enc_heads;
        let head_dim = cfg.enc_head_dim;
        let ffn_dim = cfg.enc_ffn_dim;
        let output_dim = cfg.enc_output_dim;
        let chunk_size = cfg.enc_chunk_size;
        let n_window_infer = cfg.enc_n_window_infer;

        // Determine tokens per full chunk
        let tokens_per_chunk = {
            let w = chunk_size;
            let w1 = (w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            (w2 + 2 - 3) / 2 + 1
        };

        // Transformer scratch buffers (reusable or fresh)
        let mut _owned_bufs;
        let bufs: &mut EncoderBuffers = match enc_bufs {
            Some(b) => b,
            None => {
                _owned_bufs = EncoderBuffers::new();
                &mut _owned_bufs
            }
        };

        // Conv2D stem → post-stem sequence, then size the transformer scratch.
        let (stem, total_tokens) = self.conv_stem_forward(cfg, mel, mel_frames, bufs);
        bufs.ensure(total_tokens, d_model, ffn_dim);
        let td = total_tokens * d_model;
        bufs.x[..td].copy_from_slice(&stem);

        // Build attention window boundaries
        let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
        let n_windows = total_tokens.div_ceil(window_token_size);
        bufs.window_starts.resize(n_windows + 1, 0);
        let window_starts = &mut bufs.window_starts[..n_windows + 1];
        for (w, ws) in window_starts.iter_mut().enumerate().take(n_windows) {
            *ws = (w * window_token_size) as i32;
        }
        window_starts[n_windows] = total_tokens as i32;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let tf = total_tokens * ffn_dim;

        for layer in &self.layers {
            // Self-attention
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.attn_norm_weight,
                &layer.attn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            layer.wq_weight.linear(
                &mut bufs.q[..td],
                &bufs.x_norm[..td],
                Some(&layer.wq_bias),
                total_tokens,
                d_model,
                d_model,
            );
            layer.wk_weight.linear(
                &mut bufs.k[..td],
                &bufs.x_norm[..td],
                Some(&layer.wk_bias),
                total_tokens,
                d_model,
                d_model,
            );
            layer.wv_weight.linear(
                &mut bufs.v[..td],
                &bufs.x_norm[..td],
                Some(&layer.wv_bias),
                total_tokens,
                d_model,
                d_model,
            );

            kernels::bidirectional_attention(
                &mut bufs.attn_out[..td],
                &bufs.q[..td],
                &bufs.k[..td],
                &bufs.v[..td],
                total_tokens,
                n_heads,
                head_dim,
                scale,
                &window_starts,
                n_windows,
            );

            // Fused: x += wo_bias + attn_out @ wo_weight.T
            layer.wo_weight.linear_accumulate(
                &mut bufs.x[..td],
                &bufs.attn_out[..td],
                Some(&layer.wo_bias),
                total_tokens,
                d_model,
                d_model,
            );

            // FFN
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.ffn_norm_weight,
                &layer.ffn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            layer.fc1_weight.linear(
                &mut bufs.ffn_mid[..tf],
                &bufs.x_norm[..td],
                Some(&layer.fc1_bias),
                total_tokens,
                d_model,
                ffn_dim,
            );
            kernels::gelu(&mut bufs.ffn_mid[..tf], tf);
            // Fused: x += fc2_bias + ffn_mid @ fc2_weight.T
            layer.fc2_weight.linear_accumulate(
                &mut bufs.x[..td],
                &bufs.ffn_mid[..tf],
                Some(&layer.fc2_bias),
                total_tokens,
                ffn_dim,
                d_model,
            );
        }

        // Final LayerNorm: use x_norm as temp, then swap into x
        kernels::layer_norm(
            &mut bufs.x_norm[..td],
            &bufs.x[..td],
            &self.ln_post_weight,
            &self.ln_post_bias,
            total_tokens,
            d_model,
            1e-5,
        );
        bufs.x[..td].copy_from_slice(&bufs.x_norm[..td]);

        // Projection: proj1 (GELU) -> proj2 (reuse scratch buffers)
        self.proj1_weight.linear(
            &mut bufs.q[..td],
            &bufs.x[..td],
            Some(&self.proj1_bias),
            total_tokens,
            d_model,
            d_model,
        );
        kernels::gelu(&mut bufs.q[..td], td);

        let mut enc_output = vec![0.0f32; total_tokens * output_dim];
        self.proj2_weight.linear(
            &mut enc_output,
            &bufs.q[..td],
            Some(&self.proj2_bias),
            total_tokens,
            d_model,
            output_dim,
        );

        Some((enc_output, total_tokens))
    }

    /// Batched encoder forward over K independent segments. Each `mels[i]` is
    /// `([128, frames_i], frames_i)`; returns one `(enc_output, seq)` per input,
    /// in order — identical to calling [`forward`] on each, but the per-position
    /// GEMMs (q/k/v/out projections, FFN, conv_out, proj1/2) are run ONCE over the
    /// concatenated `[ΣseqᵢΣ, d_model]` sequence. Those GEMMs are row-independent,
    /// so batching is numerically exact; the win is that the ANE then sees one
    /// large `seq` (≈ K·366) instead of K small ones — its efficient regime — and
    /// K× fewer per-call dispatches. Attention stays strictly per-segment (each
    /// segment keeps its own windows); it is a CPU kernel and never offloaded.
    ///
    /// Used by the `--mac-ane` throughput pipeline to saturate the Neural Engine.
    pub fn forward_batch(
        &self,
        cfg: &QwenConfig,
        mels: &[(&[f32], usize)],
        enc_bufs: &mut EncoderBuffers,
    ) -> Vec<(Vec<f32>, usize)> {
        let d_model = cfg.enc_d_model;
        let n_heads = cfg.enc_heads;
        let head_dim = cfg.enc_head_dim;
        let ffn_dim = cfg.enc_ffn_dim;
        let output_dim = cfg.enc_output_dim;
        let chunk_size = cfg.enc_chunk_size;
        let n_window_infer = cfg.enc_n_window_infer;
        let tokens_per_chunk = {
            let w1 = (chunk_size + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            (w2 + 2 - 3) / 2 + 1
        };
        let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);

        // Conv stem per segment → concatenate into one [S_total, d_model] sequence.
        let mut seqs = Vec::with_capacity(mels.len());
        let mut offsets = Vec::with_capacity(mels.len());
        let mut s_total = 0usize;
        let mut stems = Vec::with_capacity(mels.len());
        for &(mel, frames) in mels {
            let (stem, seq) = self.conv_stem_forward(cfg, mel, frames, enc_bufs);
            offsets.push(s_total);
            s_total += seq;
            seqs.push(seq);
            stems.push(stem);
        }
        if s_total == 0 {
            return mels.iter().map(|_| (Vec::new(), 0)).collect();
        }

        let bufs = enc_bufs;
        bufs.ensure(s_total, d_model, ffn_dim);
        let sd = s_total * d_model;
        for (i, stem) in stems.iter().enumerate() {
            let off = offsets[i] * d_model;
            bufs.x[off..off + stem.len()].copy_from_slice(stem);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let sf = s_total * ffn_dim;

        // Per-segment attention window boundaries (token indices within each segment).
        let seg_windows: Vec<Vec<i32>> = seqs
            .iter()
            .map(|&seq| {
                let n_windows = seq.div_ceil(window_token_size.max(1));
                let mut ws = vec![0i32; n_windows + 1];
                for (w, slot) in ws.iter_mut().enumerate().take(n_windows) {
                    *slot = (w * window_token_size) as i32;
                }
                ws[n_windows] = seq as i32;
                ws
            })
            .collect();

        for layer in &self.layers {
            kernels::layer_norm(
                &mut bufs.x_norm[..sd], &bufs.x[..sd],
                &layer.attn_norm_weight, &layer.attn_norm_bias, s_total, d_model, 1e-5,
            );
            // Batched q/k/v projections (the big ANE GEMMs).
            layer.wq_weight.linear(&mut bufs.q[..sd], &bufs.x_norm[..sd], Some(&layer.wq_bias), s_total, d_model, d_model);
            layer.wk_weight.linear(&mut bufs.k[..sd], &bufs.x_norm[..sd], Some(&layer.wk_bias), s_total, d_model, d_model);
            layer.wv_weight.linear(&mut bufs.v[..sd], &bufs.x_norm[..sd], Some(&layer.wv_bias), s_total, d_model, d_model);

            // Attention is per-segment (own windows); operate on each segment's slice.
            for (i, &seq) in seqs.iter().enumerate() {
                let off = offsets[i] * d_model;
                let len = seq * d_model;
                let ws = &seg_windows[i];
                let n_windows = ws.len() - 1;
                kernels::bidirectional_attention(
                    &mut bufs.attn_out[off..off + len],
                    &bufs.q[off..off + len], &bufs.k[off..off + len], &bufs.v[off..off + len],
                    seq, n_heads, head_dim, scale, ws, n_windows,
                );
            }

            // x += wo_bias + attn_out @ woᵀ (batched).
            layer.wo_weight.linear_accumulate(&mut bufs.x[..sd], &bufs.attn_out[..sd], Some(&layer.wo_bias), s_total, d_model, d_model);

            // FFN (batched).
            kernels::layer_norm(
                &mut bufs.x_norm[..sd], &bufs.x[..sd],
                &layer.ffn_norm_weight, &layer.ffn_norm_bias, s_total, d_model, 1e-5,
            );
            layer.fc1_weight.linear(&mut bufs.ffn_mid[..sf], &bufs.x_norm[..sd], Some(&layer.fc1_bias), s_total, d_model, ffn_dim);
            kernels::gelu(&mut bufs.ffn_mid[..sf], sf);
            layer.fc2_weight.linear_accumulate(&mut bufs.x[..sd], &bufs.ffn_mid[..sf], Some(&layer.fc2_bias), s_total, ffn_dim, d_model);
        }

        // Final LayerNorm + projection cascade (batched).
        kernels::layer_norm(&mut bufs.x_norm[..sd], &bufs.x[..sd], &self.ln_post_weight, &self.ln_post_bias, s_total, d_model, 1e-5);
        bufs.x[..sd].copy_from_slice(&bufs.x_norm[..sd]);
        self.proj1_weight.linear(&mut bufs.q[..sd], &bufs.x[..sd], Some(&self.proj1_bias), s_total, d_model, d_model);
        kernels::gelu(&mut bufs.q[..sd], sd);
        let mut enc_all = vec![0.0f32; s_total * output_dim];
        self.proj2_weight.linear(&mut enc_all, &bufs.q[..sd], Some(&self.proj2_bias), s_total, d_model, output_dim);

        // Split back into per-segment outputs.
        let mut out = Vec::with_capacity(seqs.len());
        for (i, &seq) in seqs.iter().enumerate() {
            let off = offsets[i] * output_dim;
            out.push((enc_all[off..off + seq * output_dim].to_vec(), seq));
        }
        out
    }
}
