# Qwen ASR Flutter - Detailed Implementation Plan

**Quick Reference Guide for Developers**

---

## Quick Start: What to Implement First

### Priority Order (Easiest → Most Impact)

```
1. Error Handling (1 day) ──────┐
2. Windows/Linux Support (2 days)├► M1: Foundation
3. Config APIs (1 day) ──────────┘

4. Streaming Core (3 days) ──────┐
5. Dart Stream API (2 days) ─────├► M2: Real-time (HIGHEST IMPACT)
6. Mic Integration (3 days) ─────┘

7. Forced Alignment (2 days) ────┐
8. Model Bundling (2 days) ──────┼► M3: Advanced
9. Export Formats (1 day) ───────┘

10. WASM Build (5 days) ─────────► M4: Web
```

---

## Task 1: Error Handling (Day 1)

### Step 1.1: Modify Rust Bridge

File: `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`

```rust
// BEFORE:
pub fn load(model_dir: String, n_threads: i32, verbosity: i32) -> Option<QwenAsrEngine>

// AFTER:
pub fn load(model_dir: String, n_threads: i32, verbosity: i32) -> Result<QwenAsrEngine, String> {
    kernels::set_verbose(verbosity);
    
    // Validate inputs
    if n_threads < 0 {
        return Err("Invalid thread count: must be >= 0".into());
    }
    
    // Check directory exists
    let path = std::path::Path::new(&model_dir);
    if !path.exists() {
        return Err(format!("Model directory not found: {}", model_dir));
    }
    
    let threads = if n_threads == 0 {
        kernels::get_num_cpus()
    } else {
        n_threads as usize
    };
    kernels::set_threads(threads);
    
    match QwenCtx::load(&model_dir) {
        Some(ctx) => Ok(QwenAsrEngine {
            inner: Mutex::new(ctx),
        }),
        None => Err(format!("Failed to load model from: {}. Ensure model*.safetensors and vocab.json exist.", model_dir)),
    }
}

// Update other methods similarly:
pub fn transcribe_file(&self, wav_path: String) -> Result<String, String> {
    if !std::path::Path::new(&wav_path).exists() {
        return Err(format!("Audio file not found: {}", wav_path));
    }
    
    let mut ctx = self.inner.lock().unwrap();
    transcribe::transcribe(&mut ctx, &wav_path)
        .ok_or_else(|| "Transcription failed - check audio format".into())
}

pub fn transcribe_pcm(&self, samples: Vec<f32>) -> Result<String, String> {
    if samples.is_empty() {
        return Err("Empty audio samples".into());
    }
    if samples.len() < 1600 { // 100ms minimum
        return Err(format!("Audio too short: {} samples (minimum 100ms = 1600 samples)", samples.len()));
    }
    
    let mut ctx = self.inner.lock().unwrap();
    transcribe::transcribe_audio(&mut ctx, &samples)
        .ok_or_else(|| "Transcription failed".into())
}
```

### Step 1.2: Update Dart Wrapper

File: `flutter/qwen_asr/lib/src/qwen_asr_engine.dart`

```dart
/// Custom exception for ASR errors.
class QAsrException implements Exception {
  final String message;
  
  const QAsrException(this.message);
  
  @override
  String toString() => 'QAsrException: $message';
}

// Update load method:
static Future<QAsrEngine> load(
  String modelDir, {
  int threads = 0,
  int verbosity = 0,
}) async {
  if (!_initialized) {
    await RustLib.init();
    _initialized = true;
  }
  
  final engine = await QwenAsrEngine.load(
    modelDir: modelDir,
    nThreads: threads,
    verbosity: verbosity,
  );
  
  if (engine == null) {
    throw const QAsrException('Failed to load model - unknown error');
  }
  
  return QAsrEngine._(engine);
}
```

### Step 1.3: Regenerate Bridge

```bash
cd flutter/qwen_asr
flutter_rust_bridge_codegen generate
```

---

## Task 2: Windows & Linux Support (Days 2-3)

### Step 2.1: Update pubspec.yaml

File: `flutter/qwen_asr/pubspec.yaml`

```yaml
flutter:
  plugin:
    platforms:
      android:
        package: com.example.qwen_asr
        pluginClass: QwenAsrPlugin
        ffiPlugin: true
      ios:
        pluginClass: QwenAsrPlugin
        ffiPlugin: true
      macos:
        pluginClass: QwenAsrPlugin
        ffiPlugin: true
      windows:
        pluginClass: QwenAsrPluginCApi
        ffiPlugin: true
      linux:
        pluginClass: QwenAsrPlugin
        ffiPlugin: true
```

### Step 2.2: Create Windows Plugin

Create: `flutter/qwen_asr/windows/CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "qwen_asr")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "qwen_asr_plugin")

add_library(${PLUGIN_NAME} SHARED
  "qwen_asr_plugin_c_api.cpp"
  "qwen_asr_plugin_c_api.h"
)

apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# Include the Rust library via cargokit
include("${CMAKE_CURRENT_SOURCE_DIR}/../cargokit/cmake/cargokit.cmake")
apply_cargokit(${PLUGIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../rust rust_lib_qwen_asr "")
```

Create: `flutter/qwen_asr/windows/qwen_asr_plugin_c_api.h`
```cpp
#ifndef FLUTTER_PLUGIN_QWEN_ASR_PLUGIN_C_API_H_
#define FLUTTER_PLUGIN_QWEN_ASR_PLUGIN_C_API_H_

#include <flutter_plugin_registrar.h>

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

FLUTTER_PLUGIN_EXPORT void QwenAsrPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#ifdef __cplusplus
}
#endif

#endif
```

Create: `flutter/qwen_asr/windows/qwen_asr_plugin_c_api.cpp`
```cpp
#include "qwen_asr_plugin_c_api.h"

// Minimal implementation - FFI handles everything
void QwenAsrPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  // No-op: using FFI
}
```

### Step 2.3: Create Linux Plugin

Create: `flutter/qwen_asr/linux/CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "qwen_asr")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "qwen_asr_plugin")

add_library(${PLUGIN_NAME} SHARED
  "qwen_asr_plugin.cc"
  "qwen_asr_plugin.h"
)

apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter PkgConfig::GTK)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED IMPORTED_TARGET gtk+-3.0)

# Include the Rust library via cargokit
include("${CMAKE_CURRENT_SOURCE_DIR}/../cargokit/cmake/cargokit.cmake")
apply_cargokit(${PLUGIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../rust rust_lib_qwen_asr "")
```

Create: `flutter/qwen_asr/linux/qwen_asr_plugin.h`
```c
#ifndef FLUTTER_PLUGIN_QWEN_ASR_PLUGIN_H_
#define FLUTTER_PLUGIN_QWEN_ASR_PLUGIN_H_

#include <flutter_linux/flutter_linux.h>

G_BEGIN_DECLS

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define FLUTTER_PLUGIN_EXPORT
#endif

gboolean qwen_asr_plugin_register_with_registrar(FlPluginRegistrar* registrar);

G_END_DECLS

#endif
```

Create: `flutter/qwen_asr/linux/qwen_asr_plugin.cc`
```cpp
#include "qwen_asr_plugin.h"

gboolean qwen_asr_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  // No-op: using FFI
  return TRUE;
}
```

### Step 2.4: Test Builds

```bash
# Windows
cd flutter/qwen_asr/example
flutter build windows --release

# Linux
cd flutter/qwen_asr/example
flutter build linux --release
```

---

## Task 3: Configuration APIs (Day 4)

### Step 3.1: Add Methods to Rust Bridge

File: `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`

```rust
/// Set an optional text prompt to guide transcription.
#[frb(sync)]
pub fn set_prompt(&self, prompt: Option<String>) {
    let mut ctx = self.inner.lock().unwrap();
    let _ = ctx.set_prompt(&prompt.unwrap_or_default());
}

/// Enable/disable silence skipping.
#[frb(sync)]
pub fn set_skip_silence(&self, skip: bool) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.skip_silence = skip;
}

/// Configure streaming chunk size in seconds.
#[frb(sync)]
pub fn set_stream_chunk_sec(&self, sec: f32) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.stream_chunk_sec = sec.clamp(0.5, 10.0);
}

/// Configure token rollback window for streaming.
#[frb(sync)]
pub fn set_stream_rollback(&self, tokens: i32) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.stream_rollback = tokens.max(0);
}

/// Configure unfixed chunks count before emitting.
#[frb(sync)]
pub fn set_stream_unfixed_chunks(&self, chunks: i32) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.stream_unfixed_chunks = chunks.max(0);
}

/// Configure max new tokens per streaming chunk.
#[frb(sync)]
pub fn set_stream_max_new_tokens(&self, tokens: i32) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.stream_max_new_tokens = tokens.max(1);
}

/// Enable past text conditioning for better context.
#[frb(sync)]
pub fn set_past_text_conditioning(&self, enable: bool) {
    let mut ctx = self.inner.lock().unwrap();
    ctx.past_text_conditioning = enable;
}

/// Get model information.
#[frb(sync)]
pub fn model_info(&self) -> ModelInfo {
    let ctx = self.inner.lock().unwrap();
    ModelInfo {
        variant: if ctx.config.dec_hidden >= 2048 { "1.7B".into() } else { "0.6B".into() },
        model_type: if ctx.config.is_aligner() { "ForcedAligner".into() } else { "ASR".into() },
        enc_hidden: ctx.config.enc_d_model as i32,
        enc_layers: ctx.config.enc_layers as i32,
        dec_hidden: ctx.config.dec_hidden as i32,
        dec_layers: ctx.config.dec_layers as i32,
    }
}

#[frb]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub variant: String,
    pub model_type: String,
    pub enc_hidden: i32,
    pub enc_layers: i32,
    pub dec_hidden: i32,
    pub dec_layers: i32,
}
```

### Step 3.2: Expose in Dart

File: `flutter/qwen_asr/lib/src/qwen_asr_engine.dart`

```dart
/// Model information.
class ModelInfo {
  final String variant;
  final String modelType;
  final int encoderHiddenSize;
  final int encoderLayers;
  final int decoderHiddenSize;
  final int decoderLayers;
  
  const ModelInfo({
    required this.variant,
    required this.modelType,
    required this.encoderHiddenSize,
    required this.encoderLayers,
    required this.decoderHiddenSize,
    required this.decoderLayers,
  });
  
  bool get isAligner => modelType == 'ForcedAligner';
  bool get isLargeModel => variant == '1.7B';
}

// Add to QAsrEngine class:

/// Set a text prompt to guide transcription.
void setPrompt(String? prompt) {
  _engine.setPrompt(prompt: prompt);
}

/// Skip silent sections in audio.
void setSkipSilence(bool skip) {
  _engine.setSkipSilence(skip: skip);
}

/// Get model information.
ModelInfo get modelInfo {
  final info = _engine.modelInfo();
  return ModelInfo(
    variant: info.variant,
    modelType: info.modelType,
    encoderHiddenSize: info.encHidden,
    encoderLayers: info.encLayers,
    decoderHiddenSize: info.decHidden,
    decoderLayers: info.decLayers,
  );
}

// Streaming config methods:
void setStreamChunkSec(double seconds) => _engine.setStreamChunkSec(sec: seconds);
void setStreamRollback(int tokens) => _engine.setStreamRollback(tokens: tokens);
void setStreamUnfixedChunks(int chunks) => _engine.setStreamUnfixedChunks(chunks: chunks);
void setStreamMaxNewTokens(int tokens) => _engine.setStreamMaxNewTokens(tokens: tokens);
void setPastTextConditioning(bool enable) => _engine.setPastTextConditioning(enable: enable);
```

---

## Task 4: Streaming Core (Days 5-7)

### Step 4.1: Create Streaming Module

Create: `flutter/qwen_asr/rust/src/api/streaming.rs`

```rust
use flutter_rust_bridge::frb;
use qwen_asr::transcribe::StreamState;
use crate::api::qwen_asr_bridge::QwenAsrEngine;
use std::sync::Mutex;

#[frb(opaque)]
pub struct QwenAsrStream {
    state: Mutex<StreamState>,
    audio_buffer: Mutex<Vec<f32>>,
}

#[frb]
impl QwenAsrStream {
    #[frb(sync)]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(StreamState::new()),
            audio_buffer: Mutex::new(Vec::with_capacity(16000 * 60)),
        }
    }
    
    #[frb(sync)]
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        let mut buffer = self.audio_buffer.lock().unwrap();
        state.reset();
        buffer.clear();
    }
    
    pub fn push_audio(&self, engine: &QwenAsrEngine, samples: Vec<f32>) -> Option<String> {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.extend_from_slice(&samples);
        
        let mut ctx = engine.inner.lock().unwrap();
        let mut state = self.state.lock().unwrap();
        
        qwen_asr::transcribe::stream_push_audio(&mut ctx, &buffer, &mut state, false)
    }
    
    pub fn finalize(&self, engine: &QwenAsrEngine) -> Option<String> {
        let buffer = self.audio_buffer.lock().unwrap();
        let mut ctx = engine.inner.lock().unwrap();
        let mut state = self.state.lock().unwrap();
        
        qwen_asr::transcribe::stream_push_audio(&mut ctx, &buffer, &mut state, true)
    }
    
    #[frb(sync)]
    pub fn text(&self) -> String {
        let state = self.state.lock().unwrap();
        state.text()
    }
    
    #[frb(sync)]
    pub fn audio_cursor_samples(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.audio_cursor()
    }
}
```

### Step 4.2: Register Module

File: `flutter/qwen_asr/rust/src/api/mod.rs`

```rust
pub mod qwen_asr_bridge;
pub mod streaming;
```

### Step 4.3: Regenerate Bridge

```bash
cd flutter/qwen_asr
flutter_rust_bridge_codegen generate
```

---

## Task 5: Dart Streaming API (Days 8-9)

Create: `flutter/qwen_asr/lib/src/qwen_asr_stream.dart`

```dart
import 'dart:async';
import 'dart:typed_data';
import 'qwen_asr_engine.dart';
import 'rust/api/streaming.dart';

class StreamingConfig {
  final double chunkSec;
  final int rollbackTokens;
  final int unfixedChunks;
  final int maxNewTokens;
  final bool pastTextConditioning;
  
  const StreamingConfig({
    this.chunkSec = 2.0,
    this.rollbackTokens = 5,
    this.unfixedChunks = 2,
    this.maxNewTokens = 32,
    this.pastTextConditioning = false,
  });
}

class QAsrStream {
  final QAsrEngine _engine;
  final QwenAsrStream _stream;
  final _textController = StreamController<String>.broadcast();
  final _deltaController = StreamController<String>.broadcast();
  var _isFinalized = false;
  var _lastEmittedText = '';
  
  QAsrStream._(this._engine, this._stream);
  
  static Future<QAsrStream> create(QAsrEngine engine, {StreamingConfig config = const StreamingConfig()}) async {
    engine
      ..setStreamChunkSec(config.chunkSec)
      ..setStreamRollback(config.rollbackTokens)
      ..setStreamUnfixedChunks(config.unfixedChunks)
      ..setStreamMaxNewTokens(config.maxNewTokens)
      ..setPastTextConditioning(config.pastTextConditioning);
    
    final stream = await QwenAsrStream.newInstance();
    return QAsrStream._(engine, stream);
  }
  
  Stream<String> get deltaStream => _deltaController.stream;
  Stream<String> get textStream => _textController.stream;
  bool get isFinalized => _isFinalized;
  String get currentText => _stream.text();
  double get processedDurationSeconds => _stream.audioCursorSamples() / 16000.0;
  
  Future<void> addAudio(Float32List samples) async {
    if (_isFinalized) throw StateError('Stream already finalized');
    
    final delta = await _stream.pushAudio(engine: _engine._engine, samples: samples.toList());
    if (delta != null && delta.isNotEmpty) _deltaController.add(delta);
    
    final fullText = _stream.text();
    if (fullText != _lastEmittedText) {
      _lastEmittedText = fullText;
      _textController.add(fullText);
    }
  }
  
  Future<String> finalize() async {
    if (_isFinalized) return _stream.text();
    _isFinalized = true;
    
    final delta = await _stream.finalize(engine: _engine._engine);
    if (delta != null && delta.isNotEmpty) _deltaController.add(delta);
    
    final result = _stream.text();
    _textController.add(result);
    await _textController.close();
    await _deltaController.close();
    return result;
  }
  
  void dispose() {
    _stream.dispose();
    if (!_textController.isClosed) _textController.close();
    if (!_deltaController.isClosed) _deltaController.close();
  }
}
```

Update `flutter/qwen_asr/lib/qwen_asr.dart`:

```dart
library;

export 'src/qwen_asr_engine.dart';
export 'src/qwen_asr_stream.dart';
```

---

## Summary

This plan provides:
1. **Week 1:** Error handling + Windows/Linux support + Config APIs
2. **Week 2:** Full streaming implementation with working example

Start with Task 1 (Error Handling) for immediate API improvement!
