# Qwen ASR Flutter Implementation Roadmap

**Version:** 1.0  
**Last Updated:** 2026-04-11  
**Estimated Total Duration:** 6-8 weeks

---

## Overview

This roadmap details the implementation plan to bring the Flutter Qwen ASR binding to production-ready status. The work is organized into 4 phases with incremental deliverables.

---

## Phase 1: Foundation & Platform Support (Week 1-2)

**Goal:** Solidify the existing API and expand platform coverage

### 1.1 Error Handling Improvement

**Task:** Replace `Option<T>` with `Result<T, String>` for better error messages

**Files to Modify:**
- `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`
- `flutter/qwen_asr/lib/src/rust/api/qwen_asr_bridge.dart` (auto-generated)
- `flutter/qwen_asr/lib/src/qwen_asr_engine.dart`

**Implementation Details:**
```rust
// BEFORE
pub fn load(model_dir: String, ...) -> Option<QwenAsrEngine>

// AFTER  
pub fn load(model_dir: String, ...) -> Result<QwenAsrEngine, String>
```

**Error Cases to Handle:**
- Model file not found
- Invalid model format
- Out of memory
- Invalid language code
- Audio format errors

**Acceptance Criteria:**
- [ ] All public APIs return `Result<T, String>`
- [ ] Dart side throws meaningful exceptions
- [ ] Error messages are user-friendly

---

### 1.2 Add Windows & Linux Platform Support

**Task:** Extend platform support beyond mobile/desktop Apple platforms

**Files to Modify:**
- `flutter/qwen_asr/pubspec.yaml`
- `flutter/qwen_asr/windows/` (new directory)
- `flutter/qwen_asr/linux/` (new directory)
- `flutter/qwen_asr/rust/Cargo.toml`

**Implementation Details:**

Add to `pubspec.yaml`:
```yaml
flutter:
  plugin:
    platforms:
      windows:
        pluginClass: QwenAsrPluginCApi
        ffiPlugin: true
      linux:
        pluginClass: QwenAsrPlugin
        ffiPlugin: true
```

Create Windows plugin stub:
```cpp
// windows/qwen_asr_plugin_c_api.cpp
#include "qwen_asr_plugin_c_api.h"

void QwenAsrPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {}
```

Create Linux plugin stub:
```cpp
// linux/qwen_asr_plugin.cc
#include "include/qwen_asr/qwen_asr_plugin.h"

void qwen_asr_plugin_register_with_registrar(FlPluginRegistrar* registrar) {}
```

**Acceptance Criteria:**
- [ ] Plugin builds on Windows (x64)
- [ ] Plugin builds on Linux (x64, ARM64)
- [ ] CI/CD updated to build all platforms
- [ ] Example app runs on Windows/Linux

---

### 1.3 Expose Missing Configuration Options

**Task:** Add missing configuration APIs from core library

**Files to Modify:**
- `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`
- `flutter/qwen_asr/lib/src/qwen_asr_engine.dart`

**New APIs to Add:**

| Method | Rust Field | Description |
|--------|-----------|-------------|
| `setPrompt(String?)` | `ctx.prompt` | Guide transcription with context |
| `setSkipSilence(bool)` | `ctx.skip_silence` | Remove silent sections |
| `setStreamChunkSec(double)` | `ctx.stream_chunk_sec` | Chunk size for streaming |
| `setStreamRollback(int)` | `ctx.stream_rollback` | Token rollback window |
| `setStreamMaxNewTokens(int)` | `ctx.stream_max_new_tokens` | Max tokens per chunk |
| `setPastTextConditioning(bool)` | `ctx.past_text_conditioning` | Use previous context |

**Implementation Pattern:**
```rust
#[frb(sync)]
pub fn set_prompt(&self, prompt: Option<String>) {
    let mut ctx = self.inner.lock().unwrap();
    let _ = ctx.set_prompt(&prompt.unwrap_or_default());
}
```

**Acceptance Criteria:**
- [ ] All 6 configuration methods implemented
- [ ] Dart wrapper exposes clean API
- [ ] Documentation updated

---

### 1.4 Memory Management Enhancement

**Task:** Add explicit disposal and resource cleanup

**Files to Modify:**
- `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`
- `flutter/qwen_asr/lib/src/qwen_asr_engine.dart`

**Implementation:**
```rust
impl Drop for QwenAsrEngine {
    fn drop(&mut self) {
        // Log cleanup for debugging
        if kernels::verbose() >= 1 {
            eprintln!("QwenAsrEngine: cleaning up resources");
        }
    }
}
```

**Acceptance Criteria:**
- [ ] No memory leaks in long-running sessions
- [ ] Valgrind/heaptrack validation on Linux

---

## Phase 2: Real-time Streaming (Week 3-4)

**Goal:** Enable real-time transcription from microphone streams

### 2.1 Core Streaming State Bridge

**Task:** Create FFI bridge for `StreamState`

**Files to Create/Modify:**
- `flutter/qwen_asr/rust/src/api/streaming.rs` (new)
- `flutter/qwen_asr/rust/src/api/mod.rs`
- `flutter/qwen_asr/rust/src/api/qwen_asr_bridge.rs`

**Implementation Details:**

```rust
// rust/src/api/streaming.rs
use flutter_rust_bridge::frb;
use qwen_asr::transcribe::StreamState;

#[frb(opaque)]
pub struct QwenAsrStream {
    state: StreamState,
    audio_buffer: Vec<f32>,
}

#[frb(opaque)]
pub struct StreamConfig {
    pub chunk_sec: f32,
    pub rollback: i32,
    pub unfixed_chunks: i32,
    pub max_new_tokens: i32,
}

impl QwenAsrStream {
    #[frb(sync)]
    pub fn new() -> Self {
        Self {
            state: StreamState::new(),
            audio_buffer: Vec::with_capacity(16000 * 30), // 30s pre-allocated
        }
    }
    
    /// Push audio and get incremental text delta
    pub fn push_audio(&mut self, engine: &QwenAsrEngine, samples: Vec<f32>) -> Option<String> {
        self.audio_buffer.extend_from_slice(&samples);
        let mut ctx = engine.inner.lock().unwrap();
        qwen_asr::transcribe::stream_push_audio(
            &mut ctx,
            &self.audio_buffer,
            &mut self.state,
            false, // not finalizing
        )
    }
    
    /// Finalize stream and flush remaining tokens
    pub fn finalize(&mut self, engine: &QwenAsrEngine) -> Option<String> {
        let mut ctx = engine.inner.lock().unwrap();
        qwen_asr::transcribe::stream_push_audio(
            &mut ctx,
            &self.audio_buffer,
            &mut self.state,
            true, // finalize
        )
    }
    
    /// Reset for new utterance
    #[frb(sync)]
    pub fn reset(&mut self) {
        self.state.reset();
        self.audio_buffer.clear();
    }
    
    /// Get full accumulated text so far
    #[frb(sync)]
    pub fn text(&self) -> String {
        self.state.text()
    }
    
    /// Get audio cursor position
    #[frb(sync)]
    pub fn audio_cursor_samples(&self) -> usize {
        self.state.audio_cursor()
    }
}
```

**Acceptance Criteria:**
- [ ] Stream state properly managed across FFI boundary
- [ ] No audio sample loss
- [ ] Thread-safe access to engine

---

### 2.2 Dart Streaming API

**Task:** Create high-level Dart API for streaming

**Files to Create:**
- `flutter/qwen_asr/lib/src/qwen_asr_stream.dart`
- Update `flutter/qwen_asr/lib/qwen_asr.dart`

**Implementation Details:**

```dart
// lib/src/qwen_asr_stream.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'qwen_asr_engine.dart';
import 'rust/api/streaming.dart' show QwenAsrStream;

/// Configuration for streaming transcription.
class StreamingConfig {
  final double chunkSec;
  final int rollbackTokens;
  final int unfixedChunks;
  final int maxNewTokens;
  
  const StreamingConfig({
    this.chunkSec = 2.0,
    this.rollbackTokens = 5,
    this.unfixedChunks = 2,
    this.maxNewTokens = 32,
  });
}

/// High-level streaming transcription API.
/// 
/// ```dart
/// final stream = await engine.createStream();
/// stream.textStream.listen(print);
/// stream.addAudio(pcmChunk);
/// stream.finalize();
/// ```
class QAsrStream {
  final QAsrEngine _engine;
  final QwenAsrStream _stream;
  final _textController = StreamController<String>.broadcast();
  final _deltaController = StreamController<String>.broadcast();
  
  bool _isFinalized = false;
  
  QAsrStream._(this._engine, this._stream);
  
  /// Create a new streaming session.
  static Future<QAsrStream> create(
    QAsrEngine engine, {
    StreamingConfig config = const StreamingConfig(),
  }) async {
    // Apply config to engine context
    engine.setStreamChunkSec(config.chunkSec);
    engine.setStreamRollback(config.rollbackTokens);
    engine.setStreamUnfixedChunks(config.unfixedChunks);
    engine.setStreamMaxNewTokens(config.maxNewTokens);
    
    final stream = await QwenAsrStream.newInstance();
    return QAsrStream._(engine, stream);
  }
  
  /// Stream of incremental text deltas (new words as they arrive).
  Stream<String> get deltaStream => _deltaController.stream;
  
  /// Stream of full accumulated text after each chunk.
  Stream<String> get textStream => _textController.stream;
  
  /// Add PCM audio samples (f32, 16kHz, mono).
  /// Can be called multiple times as audio arrives.
  Future<void> addAudio(Float32List samples) async {
    if (_isFinalized) {
      throw StateError('Stream already finalized');
    }
    
    final delta = await _stream.pushAudio(
      engine: _engine._engine,
      samples: samples.toList(),
    );
    
    if (delta != null && delta.isNotEmpty) {
      _deltaController.add(delta);
      _textController.add(_stream.text());
    }
  }
  
  /// Finalize stream and emit remaining tokens.
  Future<String> finalize() async {
    if (_isFinalized) return _stream.text();
    _isFinalized = true;
    
    final delta = await _stream.finalize(engine: _engine._engine);
    if (delta != null && delta.isNotEmpty) {
      _deltaController.add(delta);
    }
    
    final result = _stream.text();
    _textController.add(result);
    await _textController.close();
    await _deltaController.close();
    
    return result;
  }
  
  /// Reset for new utterance without recreating stream.
  void reset() {
    _isFinalized = false;
    _stream.reset();
  }
  
  /// Get current accumulated text.
  String get currentText => _stream.text();
  
  /// Get processed audio duration in seconds.
  double get processedDurationSeconds => 
      _stream.audioCursorSamples() / 16000.0;
  
  /// Dispose resources.
  void dispose() {
    _stream.dispose();
    if (!_textController.isClosed) _textController.close();
    if (!_deltaController.isClosed) _deltaController.close();
  }
}
```

**Acceptance Criteria:**
- [ ] Stream API is easy to use
- [ ] Memory efficient for long sessions
- [ ] Proper cleanup on dispose

---

### 2.3 Microphone Integration Helper

**Task:** Create helper for Flutter microphone integration

**Files to Create:**
- `flutter/qwen_asr/lib/src/audio/microphone_recorder.dart`
- `flutter/qwen_asr/lib/src/audio/audio_buffer.dart`

**Implementation Details:**

```dart
// lib/src/audio/microphone_recorder.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Audio format for microphone input.
class AudioFormat {
  final int sampleRate;
  final int channels;
  
  const AudioFormat({
    this.sampleRate = 16000,
    this.channels = 1,
  });
}

/// Abstract microphone recorder interface.
abstract class MicrophoneRecorder {
  /// Stream of audio chunks as they arrive.
  Stream<Float32List> get audioStream;
  
  /// Start recording.
  Future<void> start({AudioFormat format = const AudioFormat()});
  
  /// Stop recording.
  Future<void> stop();
  
  /// Dispose resources.
  void dispose();
}

/// Factory to create platform-specific recorder.
/// Uses method channel for native implementation.
class MicrophoneRecorderFactory {
  static const _channel = MethodChannel('qwen_asr/microphone');
  
  static Future<MicrophoneRecorder> create() async {
    // Platform-specific implementations
    if (Platform.isAndroid || Platform.isIOS) {
      return _MobileMicrophoneRecorder(_channel);
    }
    throw UnsupportedError('Platform not supported');
  }
}
```

**Platform Implementation (Android):**

```kotlin
// android/src/main/kotlin/.../MicrophoneHandler.kt
class MicrophoneHandler(private val context: Context) {
    private var audioRecord: AudioRecord? = null
    private val audioBufferSize = AudioRecord.getMinBufferSize(
        16000,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT
    )
    
    fun startRecording(eventSink: EventChannel.EventSink?) {
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            16000,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
            audioBufferSize
        )
        audioRecord?.startRecording()
        
        // Start reading thread
        thread {
            val buffer = FloatArray(1600) // 100ms chunks
            while (audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val read = audioRecord?.read(buffer, 0, buffer.size, 
                    AudioRecord.READ_BLOCKING) ?: 0
                if (read > 0) {
                    val chunk = buffer.copyOf(read)
                    // Send to Flutter via EventChannel
                    activity.runOnUiThread {
                        eventSink?.success(chunk.toList())
                    }
                }
            }
        }
    }
}
```

**Acceptance Criteria:**
- [ ] Works on Android and iOS
- [ ] 16kHz mono f32 PCM output
- [ ] Low latency (< 200ms)

---

### 2.4 Example App Streaming Demo

**Task:** Update example app with real-time transcription

**Files to Modify:**
- `flutter/qwen_asr/example/lib/main.dart`

**Features to Add:**
- Live microphone transcription
- Real-time text display with delta highlighting
- Recording controls (start/stop)
- Visual audio waveform

**Acceptance Criteria:**
- [ ] One-tap live transcription
- [ ] Text appears in real-time
- [ ] Handles long sessions (10+ minutes)

---

## Phase 3: Forced Alignment & Advanced Features (Week 5-6)

**Goal:** Add subtitle/word-level timestamp support

### 3.1 Forced Alignment Bridge

**Task:** Expose `forced_align` function

**Files to Create:**
- `flutter/qwen_asr/rust/src/api/alignment.rs` (new)

**Implementation:**

```rust
// rust/src/api/alignment.rs
use flutter_rust_bridge::frb;
use qwen_asr::align::{AlignResult, forced_align};

#[frb(mirror(AlignResult))]
pub struct _AlignResult {
    pub text: String,
    pub start_ms: f32,
    pub end_ms: f32,
}

impl QwenAsrEngine {
    /// Align words in text to audio timestamps.
    /// Requires an aligner model (qwen3-aligner-0.6b).
    pub fn align_words(
        &self,
        samples: Vec<f32>,
        text: String,
        language: String,
    ) -> Result<Vec<AlignResult>, String> {
        let mut ctx = self.inner.lock().unwrap();
        
        if !ctx.config.is_aligner() {
            return Err("Model is not a forced aligner".into());
        }
        
        forced_align(&mut ctx, &samples, &text, &language)
            .ok_or_else(|| "Alignment failed".into())
    }
}
```

**Acceptance Criteria:**
- [ ] Word-level timestamps accurate to ~100ms
- [ ] Works for English and CJK languages
- [ ] Returns error for non-aligner models

---

### 3.2 Dart Alignment API

**Task:** Create Dart API for alignment

```dart
// lib/src/alignment_result.dart
class WordAlignment {
  final String text;
  final Duration start;
  final Duration end;
  
  const WordAlignment({
    required this.text,
    required this.start,
    required this.end,
  });
  
  /// Export to SRT format.
  String toSrt(int index) {
    return '''
$index
${_formatSrtTime(start)} --> ${_formatSrtTime(end)}
$text
'''.trim();
  }
  
  /// Export to WebVTT format.
  String toWebVtt() => '${_formatVttTime(start)} --> ${_formatVttTime(end)}\n$text';
}

class AlignmentResult {
  final List<WordAlignment> words;
  final String language;
  
  const AlignmentResult({required this.words, required this.language});
  
  /// Export full result to SRT format.
  String toSrt() {
    return words.asMap().entries
        .map((e) => e.value.toSrt(e.key + 1))
        .join('\n\n');
  }
  
  /// Merge words into sentence-level chunks.
  List<WordAlignment> toSentences() { ... }
}
```

**Acceptance Criteria:**
- [ ] SRT/WebVTT export works
- [ ] Sentence-level grouping available

---

### 3.3 Model Asset Bundling Helper

**Task:** Simplify model distribution via Flutter assets

**Files to Create:**
- `flutter/qwen_asr/lib/src/model_manager.dart`

**Implementation:**

```dart
// lib/src/model_manager.dart
class ModelManager {
  static Future<String> extractModelFromAssets(
    String assetPath, {
    String? modelName,
  }) async {
    final modelDir = await _getModelDirectory();
    final targetDir = Directory('${modelDir.path}/$modelName');
    
    if (await targetDir.exists()) {
      return targetDir.path; // Already extracted
    }
    
    await targetDir.create(recursive: true);
    
    // Extract asset bundle
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final modelAssets = manifest
        .listAssets()
        .where((path) => path.startsWith(assetPath));
    
    for (final asset in modelAssets) {
      final data = await rootBundle.load(asset);
      final fileName = asset.split('/').last;
      final file = File('${targetDir.path}/$fileName');
      await file.writeAsBytes(data.buffer.asUint8List());
    }
    
    return targetDir.path;
  }
  
  static Future<Directory> _getModelDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    return Directory('${appDir.path}/qwen_asr_models');
  }
  
  /// Clean up extracted models.
  static Future<void> clearCache() async {
    final dir = await _getModelDirectory();
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }
}
```

**Usage:**
```dart
// pubspec.yaml
flutter:
  assets:
    - assets/models/qwen3-asr-0.6b/

// App code
final modelPath = await ModelManager.extractModelFromAssets(
  'assets/models/qwen3-asr-0.6b',
  modelName: 'qwen3-asr-0.6b',
);
final engine = await QAsrEngine.load(modelPath);
```

**Acceptance Criteria:**
- [ ] Models extract on first run
- [ ] Cached for subsequent runs
- [ ] Compression-aware (handles .gz assets)

---

## Phase 4: Web (WASM) Support (Week 7-8)

**Goal:** Enable browser-based transcription

### 4.1 WASM Build Configuration

**Task:** Configure Rust for WASM32 target

**Files to Create/Modify:**
- `flutter/qwen_asr/rust/Cargo.toml` (add wasm feature)
- `flutter/qwen_asr/web/` (new directory)

**Cargo.toml Changes:**
```toml
[features]
default = ["mobile"]
mobile = ["flutter_rust_bridge/dart-opaque"]
wasm = ["flutter_rust_bridge/wasm"]

[dependencies]
flutter_rust_bridge = { version = "2.11.1", default-features = false }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }

# Replace qwen_asr with wasm-compatible fork
qwen_asr = { package = "qwen-asr", path = "../../../crates/qwen-asr", default-features = false, features = ["wasm"] }
```

**Acceptance Criteria:**
- [ ] `wasm-pack build` succeeds
- [ ] No SIMD dependencies in WASM build
- [ ] File size < 50MB (compressed)

---

### 4.2 Web Audio Integration

**Task:** Implement web audio capture

**Files:**
- `flutter/qwen_asr/web/audio_worklet.js`
- `flutter/qwen_asr/lib/src/audio/web_recorder.dart`

**Implementation:**

```javascript
// web/audio_worklet.js
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(1600); // 100ms @ 16kHz
    this.bufferIndex = 0;
  }
  
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length === 0) return true;
    
    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.bufferIndex++] = channel[i];
      
      if (this.bufferIndex >= this.buffer.length) {
        this.port.postMessage(this.buffer.slice());
        this.bufferIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor('audio-capture', AudioCaptureProcessor);
```

**Acceptance Criteria:**
- [ ] Works in Chrome, Firefox, Safari
- [ ] 16kHz resampling handled
- [ ] Permission handling for microphone

---

### 4.3 Model Loading for Web

**Task:** Implement model download/caching for web

```dart
// web uses IndexedDB for model storage
class WebModelCache {
  static Future<void> downloadModel(String url, String name) async {
    // Download with progress
    // Store in IndexedDB
  }
  
  static Future<Uint8List?> getFile(String modelName, String fileName) async {
    // Retrieve from IndexedDB
  }
}
```

**Acceptance Criteria:**
- [ ] Progressive download with progress bar
- [ ] Resume interrupted downloads
- [ ] Cache persists across sessions

---

## Testing Strategy

### Unit Tests
```dart
// test/qwen_asr_test.dart
group('QAsrEngine', () {
  test('load model', () async {
    final engine = await QAsrEngine.load('test/fixtures/model');
    expect(engine, isNotNull);
    addTearDown(() => engine.dispose());
  });
  
  test('transcribe audio', () async {
    final engine = await QAsrEngine.load('test/fixtures/model');
    final text = await engine.transcribeFile('test/fixtures/audio.wav');
    expect(text, contains('hello'));
  });
});

group('Streaming', () {
  test('incremental transcription', () async {
    final engine = await QAsrEngine.load('test/fixtures/model');
    final stream = await QAsrStream.create(engine);
    
    final deltas = <String>[];
    stream.deltaStream.listen(deltas.add);
    
    await stream.addAudio(Float32List.fromList([...]));
    await stream.finalize();
    
    expect(deltas, isNotEmpty);
  });
});
```

### Integration Tests
- Full transcription pipeline on each platform
- Memory leak tests (1-hour continuous streaming)
- Model loading/unloading stress test

### Benchmarks
- RTF (Real-Time Factor) measurement
- Memory usage profiling
- Battery impact on mobile

---

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/flutter.yml
name: Flutter CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Build Rust
        run: cd rust && cargo build --release
        
      - name: Flutter Test
        run: flutter test
        
  build:
    strategy:
      matrix:
        target: [android, ios, macos, windows, linux, web]
    runs-on: ${{ matrix.os }}
    steps:
      - name: Build ${{ matrix.target }}
        run: flutter build ${{ matrix.target }} --release
```

---

## Documentation

### API Documentation
- Dartdoc comments for all public APIs
- Usage examples for each feature
- Migration guide from v0.3 to v1.0

### User Guide
- "Getting Started" tutorial
- Platform-specific setup instructions
- Performance tuning guide
- Troubleshooting FAQ

---

## Milestones & Deliverables

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| M1 | Week 2 | Foundation complete - Error handling, Windows/Linux support, config APIs |
| M2 | Week 4 | Streaming complete - Real-time transcription, microphone integration |
| M3 | Week 6 | Advanced features - Alignment, model bundling, SRT export |
| M4 | Week 8 | Web support - WASM build, browser transcription |
| RC | Week 9 | Release candidate - All tests passing, docs complete |
| v1.0 | Week 10 | Production release |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| WASM performance poor | High | Keep native mobile as primary, WASM as experimental |
| Memory leaks in streaming | High | Extensive profiling, automated leak detection |
| Platform build issues | Medium | CI builds on all platforms from day 1 |
| Model size too large | Medium | Document compression options, provide model pruning guide |

---

## Appendix: File Structure

```
flutter/qwen_asr/
├── lib/
│   ├── qwen_asr.dart
│   └── src/
│       ├── qwen_asr_engine.dart
│       ├── qwen_asr_stream.dart          # NEW
│       ├── alignment_result.dart          # NEW
│       ├── model_manager.dart             # NEW
│       ├── audio/
│       │   ├── microphone_recorder.dart   # NEW
│       │   ├── web_recorder.dart          # NEW
│       │   └── audio_buffer.dart          # NEW
│       └── rust/
│           ├── api/
│           │   └── qwen_asr_bridge.dart
│           └── frb_generated.dart
├── rust/
│   └── src/
│       ├── api/
│       │   ├── mod.rs
│       │   ├── qwen_asr_bridge.rs        # MODIFIED
│       │   ├── streaming.rs              # NEW
│       │   └── alignment.rs              # NEW
│       └── lib.rs
├── web/
│   └── audio_worklet.js                  # NEW
├── example/
│   └── lib/
│       └── main.dart                     # MODIFIED
└── ROADMAP.md                            # THIS FILE
```

---

**Next Steps:**
1. Review and approve roadmap
2. Set up CI/CD for all platforms
3. Begin Phase 1 implementation
4. Weekly progress reviews
