import 'dart:async';
import 'dart:typed_data';
import 'qwen_asr_engine.dart';
import 'rust/api/streaming.dart';

/// Configuration for streaming transcription.
///
/// All parameters have sensible defaults for real-time transcription.
class StreamingConfig {
  /// Audio chunk size in seconds.
  ///
  /// Smaller values = lower latency but more processing overhead.
  /// Valid range: 0.5 to 10.0. Default: 2.0
  final double chunkSec;

  /// Token rollback window for stability.
  ///
  /// Number of tokens to keep as "unstable" at the end of each chunk.
  /// These may change as more audio arrives. Default: 5
  final int rollbackTokens;

  /// Number of chunks to wait before emitting text.
  ///
  /// Initial chunks are held back to allow context to build.
  /// Default: 2
  final int unfixedChunks;

  /// Maximum new tokens to generate per chunk.
  ///
  /// Prevents runaway generation on ambiguous audio.
  /// Default: 32
  final int maxNewTokens;

  /// Use past text conditioning for better context.
  ///
  /// When enabled, previous transcription context is used for each chunk.
  /// May improve accuracy but increases latency. Default: false
  final bool pastTextConditioning;

  const StreamingConfig({
    this.chunkSec = 2.0,
    this.rollbackTokens = 5,
    this.unfixedChunks = 2,
    this.maxNewTokens = 32,
    this.pastTextConditioning = false,
  });

  /// Low-latency configuration for real-time captions.
  static const lowLatency = StreamingConfig(
    chunkSec: 0.5,
    rollbackTokens: 3,
    unfixedChunks: 1,
    maxNewTokens: 16,
  );

  /// High-accuracy configuration for offline processing.
  static const highAccuracy = StreamingConfig(
    chunkSec: 3.0,
    rollbackTokens: 8,
    unfixedChunks: 3,
    maxNewTokens: 64,
    pastTextConditioning: true,
  );
}

/// High-level streaming transcription API for real-time audio.
///
/// Create a stream, feed audio chunks as they arrive, and listen to
/// the text output in real-time.
///
/// ```dart
/// final engine = await QAsrEngine.load(modelPath);
/// final stream = await QAsrStream.create(engine);
///
/// // Listen to incremental results
/// stream.deltaStream.listen((delta) => print('New: $delta'));
/// stream.textStream.listen((text) => print('Total: $text'));
///
/// // Feed audio from microphone
/// microphone.onAudio.listen((chunk) => stream.addAudio(chunk));
///
/// // Finalize when done
/// final result = await stream.finalize();
/// ```
class QAsrStream {
  final QAsrEngine _engine;
  final QwenAsrStream _stream;
  final _textController = StreamController<String>.broadcast();
  final _deltaController = StreamController<String>.broadcast();
  bool _isFinalized = false;
  bool _isDisposed = false;
  String _lastEmittedText = '';

  QAsrStream._(this._engine, this._stream);

  /// Create a new streaming session with the given engine.
  ///
  /// Applies [config] to the engine before starting.
  static Future<QAsrStream> create(
    QAsrEngine engine, {
    StreamingConfig config = const StreamingConfig(),
  }) async {
    // Apply configuration to engine
    engine
      ..setStreamChunkSec(config.chunkSec)
      ..setStreamRollback(config.rollbackTokens)
      ..setStreamUnfixedChunks(config.unfixedChunks)
      ..setStreamMaxNewTokens(config.maxNewTokens)
      ..setPastTextConditioning(config.pastTextConditioning);

    final stream = QwenAsrStream();
    return QAsrStream._(engine, stream);
  }

  /// Stream of incremental text deltas (new words as they arrive).
  ///
  /// Each event contains only the newly recognized text since the last event.
  Stream<String> get deltaStream => _deltaController.stream;

  /// Stream of full accumulated text after each processing chunk.
  ///
  /// Each event contains the complete transcription so far.
  Stream<String> get textStream => _textController.stream;

  /// Whether the stream has been finalized.
  bool get isFinalized => _isFinalized;

  /// Whether the stream has been disposed.
  bool get isDisposed => _isDisposed;

  /// Get the current accumulated text without finalizing.
  ///
  /// This may include unstable tokens that could change.
  String get currentText {
    _checkDisposed();
    return _stream.text();
  }

  /// Get the duration of processed audio in seconds.
  double get processedDurationSeconds {
    _checkDisposed();
    return _stream.processedSeconds();
  }

  /// Add PCM audio samples (f32, 16kHz, mono).
  ///
  /// Can be called multiple times as audio arrives from the microphone.
  /// Samples are accumulated internally.
  ///
  /// Throws [StateError] if called after [finalize] or on a disposed stream.
  Future<void> addAudio(Float32List samples) async {
    _checkDisposed();
    if (_isFinalized) {
      throw StateError('Cannot add audio to finalized stream');
    }

    final delta = await _stream.pushAudio(
      engine: _engine.internalEngine,
      samples: samples.toList(),
    );

    // Emit delta if we got new text
    if (delta != null && delta.isNotEmpty) {
      _deltaController.add(delta);
    }

    // Always emit full text for consistency
    final fullText = _stream.text();
    if (fullText != _lastEmittedText) {
      _lastEmittedText = fullText;
      _textController.add(fullText);
    }
  }

  /// Finalize the stream and emit any remaining tokens.
  ///
  /// Call this when the audio stream has ended (e.g., user stopped recording).
  /// Returns the complete transcription result.
  ///
  /// Can only be called once. Subsequent calls return the cached result.
  ///
  /// Automatically disposes the stream - do not call [dispose] after this.
  Future<String> finalize() async {
    _checkDisposed();
    if (_isFinalized) {
      return _stream.text();
    }

    _isFinalized = true;

    final delta = await _stream.finalize(engine: _engine.internalEngine);
    if (delta != null && delta.isNotEmpty) {
      _deltaController.add(delta);
    }

    final result = _stream.text();
    _textController.add(result);

    // Close streams
    await _textController.close();
    await _deltaController.close();

    return result;
  }

  /// Reset for a new utterance without recreating the stream.
  ///
  /// Clears accumulated audio and resets state. Can be used for
  /// phrase-level transcription (e.g., voice commands).
  ///
  /// Cannot be called after [finalize].
  void reset() {
    _checkDisposed();
    if (_isFinalized) {
      throw StateError('Cannot reset finalized stream');
    }

    _stream.reset();
    _lastEmittedText = '';
  }

  /// Dispose resources.
  ///
  /// Call this when done with the stream. If [finalize] was called,
  /// this is a no-op (finalize already disposes).
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;

    _stream.dispose();

    if (!_textController.isClosed) {
      _textController.close();
    }
    if (!_deltaController.isClosed) {
      _deltaController.close();
    }
  }

  void _checkDisposed() {
    if (_isDisposed) {
      throw StateError('Stream has been disposed');
    }
  }
}
