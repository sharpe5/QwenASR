import 'dart:typed_data';
import 'package:flutter/foundation.dart' show internal;
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'rust/api/qwen_asr_bridge.dart';
import 'rust/api/streaming.dart' show ModelInfo;
import 'rust/api/alignment.dart' as rust;
import 'rust/frb_generated.dart';

/// Exception thrown when ASR operations fail.
class QAsrException implements Exception {
  final String message;

  const QAsrException(this.message);

  @override
  String toString() => 'QAsrException: $message';
}

/// Information about a loaded model.
class QAsrModelInfo {
  final String variant;
  final String modelType;
  final int encoderHiddenSize;
  final int encoderLayers;
  final int decoderHiddenSize;
  final int decoderLayers;

  const QAsrModelInfo({
    required this.variant,
    required this.modelType,
    required this.encoderHiddenSize,
    required this.encoderLayers,
    required this.decoderHiddenSize,
    required this.decoderLayers,
  });

  /// Whether this is an aligner model (for word-level timestamps).
  bool get isAligner => modelType == 'ForcedAligner';

  /// Whether this is the larger 1.7B model.
  bool get isLargeModel => variant == '1.7B';

  /// Whether this is the smaller 0.6B model.
  bool get isSmallModel => variant == '0.6B';

  @override
  String toString() =>
      'QAsrModelInfo(variant: $variant, type: $modelType, encoder: ${encoderHiddenSize}x$encoderLayers, decoder: ${decoderHiddenSize}x$decoderLayers)';
}

/// On-device speech-to-text engine powered by Qwen3-ASR.
///
/// Load a model with [load], transcribe audio with [transcribeFile]
/// or [transcribePcm], and call [dispose] when done.
///
/// ```dart
/// final engine = await QAsrEngine.load('/path/to/model');
/// final text = await engine.transcribeFile('audio.wav');
/// engine.dispose();
/// ```
class QAsrEngine {
  final QwenAsrEngine _engine;
  QAsrEngine._(this._engine);

  static bool _initialized = false;

  /// Initialize the Rust library with a custom dylib path.
  /// Call this before [load] when running outside a Flutter app
  /// (e.g. in `flutter test`).
  static Future<void> initWith({required String dylibPath}) async {
    if (_initialized) return;
    await RustLib.init(externalLibrary: ExternalLibrary.open(dylibPath));
    _initialized = true;
  }

  /// Load a Qwen3-ASR model from [modelDir].
  ///
  /// The directory must contain `model*.safetensors` and `vocab.json`.
  /// Set [threads] to control parallelism (0 = auto-detect CPU count) and
  /// [verbosity] for logging (0 = silent, 1 = info, 2 = debug).
  ///
  /// Throws [QAsrException] if the model cannot be loaded.
  static Future<QAsrEngine> load(
    String modelDir, {
    int threads = 0,
    int verbosity = 0,
  }) async {
    if (!_initialized) {
      await RustLib.init();
      _initialized = true;
    }

    final result = await QwenAsrEngine.load(
      modelDir: modelDir,
      nThreads: threads,
      verbosity: verbosity,
    );

    // Handle potential error
    try {
      final engine = result;
      if (engine == null) {
        throw const QAsrException('Failed to load model - returned null');
      }
      return QAsrEngine._(engine);
    } catch (e) {
      throw QAsrException('Failed to load model: $e');
    }
  }

  /// Transcribe a WAV file at [wavPath].
  ///
  /// Returns the transcribed text, or empty string on failure.
  /// Throws [QAsrException] if the file doesn't exist.
  Future<String> transcribeFile(String wavPath) async {
    final result = await _engine.transcribeFile(wavPath: wavPath);
    return result;
  }

  /// Transcribe raw PCM audio.
  ///
  /// [samples] must be a [Float32List] of 16 kHz mono audio with values
  /// normalized to the range -1.0 to 1.0.
  ///
  /// Minimum length: 100ms (1600 samples at 16kHz).
  Future<String> transcribePcm(Float32List samples) async {
    final result = await _engine.transcribePcm(samples: samples.toList());
    return result;
  }

  /// Transcribe from a WAV file buffer (bytes).
  Future<String> transcribeWavBuffer(Uint8List wavData) async {
    final result = await _engine.transcribeWavBuffer(wavData: wavData.toList());
    return result;
  }

  /// Set segment duration in seconds for splitting long audio.
  ///
  /// Use 30.0 as a good default for long recordings. Set to 0 to disable
  /// segmentation (transcribe the entire file in one pass).
  void setSegmentSec(double sec) {
    _engine.setSegmentSec(sec: sec);
  }

  /// Force a specific language (e.g. `"English"`, `"Chinese"`, `"Japanese"`).
  ///
  /// Pass an empty string to return to auto-detection. Returns `true` if the
  /// language name was recognized, `false` otherwise.
  bool setLanguage(String language) {
    return _engine.setLanguage(language: language);
  }

  /// Get performance stats from last transcription.
  String perfStats() {
    return _engine.perfStats();
  }

  // ==================== Configuration APIs ====================

  /// Set a text prompt to guide transcription (e.g., "medical terminology").
  ///
  /// Pass null or empty string to clear.
  void setPrompt(String? prompt) {
    _engine.setPrompt(prompt: prompt);
  }

  /// Skip silent sections in audio (useful for long recordings).
  void setSkipSilence(bool skip) {
    _engine.setSkipSilence(skip: skip);
  }

  /// Get model information.
  QAsrModelInfo get modelInfo {
    final info = _engine.modelInfo();
    return QAsrModelInfo(
      variant: info.variant,
      modelType: info.modelType,
      encoderHiddenSize: info.encHidden,
      encoderLayers: info.encLayers,
      decoderHiddenSize: info.decHidden,
      decoderLayers: info.decLayers,
    );
  }

  /// Configure streaming chunk size in seconds (default: 2.0).
  void setStreamChunkSec(double seconds) {
    _engine.setStreamChunkSec(sec: seconds);
  }

  /// Configure token rollback window for streaming (default: 5).
  void setStreamRollback(int tokens) {
    _engine.setStreamRollback(tokens: tokens);
  }

  /// Configure unfixed chunks count before emitting (default: 2).
  void setStreamUnfixedChunks(int chunks) {
    _engine.setStreamUnfixedChunks(chunks: chunks);
  }

  /// Configure max new tokens per streaming chunk (default: 32).
  void setStreamMaxNewTokens(int tokens) {
    _engine.setStreamMaxNewTokens(tokens: tokens);
  }

  /// Enable past text conditioning for better streaming context.
  void setPastTextConditioning(bool enable) {
    _engine.setPastTextConditioning(enable: enable);
  }

  /// Dispose the engine and free resources.
  void dispose() {
    _engine.dispose();
  }

  // ==================== Internal Access (for extensions) ====================

  /// Internal engine access for extensions.
  @internal
  QwenAsrEngine get internalEngine => _engine;

  /// Check if the loaded model is a forced aligner.
  bool get isAligner => _engine.isAligner();

  /// Perform forced alignment on audio with a known transcript.
  ///
  /// Returns word-level timestamps showing when each word was spoken.
  /// Requires an aligner model (qwen3-aligner-0.6b).
  ///
  /// [samples] must be f32 PCM at 16kHz, mono.
  /// [text] is the known transcript to align.
  /// [language] controls word splitting - use "Chinese", "Japanese",
  /// "Korean", "Cantonese" for character-level, anything else for word-level.
  ///
  /// Throws [StateError] if the model is not an aligner.
  /// Throws [QAsrException] if alignment fails.
  Future<rust.AlignmentResult> alignWords(
    Float32List samples,
    String text, {
    String language = "English",
  }) async {
    if (!isAligner) {
      throw StateError(
        'Model is not a forced aligner. Use qwen3-aligner-0.6b model.',
      );
    }

    if (text.isEmpty) {
      throw ArgumentError('Transcript cannot be empty');
    }

    try {
      final result = await _engine.alignWords(
        samples: samples.toList(),
        text: text,
        language: language,
      );
      return result;
    } catch (e) {
      throw QAsrException('Alignment failed: $e');
    }
  }
}
