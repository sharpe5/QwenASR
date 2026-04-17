import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Audio format specification for recording.
class AudioFormat {
  /// Sample rate in Hz. Default: 16000 (16kHz) - required by Qwen ASR.
  final int sampleRate;

  /// Number of channels. Default: 1 (mono) - required by Qwen ASR.
  final int channels;

  /// Bits per sample. Default: 32 (float32).
  final int bitsPerSample;

  const AudioFormat({
    this.sampleRate = 16000,
    this.channels = 1,
    this.bitsPerSample = 32,
  });

  /// Standard format for Qwen ASR: 16kHz, mono, float32.
  static const qwenAsr = AudioFormat(
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 32,
  );

  /// Get bytes per sample.
  int get bytesPerSample => bitsPerSample ~/ 8;

  /// Get bytes per frame (sample * channels).
  int get bytesPerFrame => bytesPerSample * channels;
}

/// State of the microphone recorder.
enum RecorderState {
  /// Recorder is initialized but not recording.
  initialized,

  /// Recorder is currently recording audio.
  recording,

  /// Recorder has been stopped.
  stopped,

  /// Recorder encountered an error.
  error,
}

/// Abstract interface for microphone recording.
///
/// Use [MicrophoneRecorder.create] to get a platform-specific implementation.
abstract class MicrophoneRecorder {
  /// Stream of audio chunks as they arrive.
  ///
  /// Each chunk is a [Float32List] of PCM samples at the configured sample rate.
  Stream<Float32List> get audioStream;

  /// Current state of the recorder.
  RecorderState get state;

  /// Whether the recorder is currently recording.
  bool get isRecording;

  /// Start recording audio.
  ///
  /// [format] specifies the audio format. Use [AudioFormat.qwenAsr] for
  /// Qwen ASR compatibility.
  ///
  /// Throws [StateError] if already recording.
  /// Throws [PlatformException] if microphone permission is denied.
  Future<void> start({AudioFormat format = AudioFormat.qwenAsr});

  /// Stop recording audio.
  ///
  /// Returns a Future that completes when recording has stopped.
  Future<void> stop();

  /// Dispose resources.
  ///
  /// Call this when done with the recorder. Stops recording if active.
  Future<void> dispose();

  /// Create a platform-specific microphone recorder.
  ///
  /// Uses method channels to communicate with native implementations.
  factory MicrophoneRecorder.create() = _MethodChannelMicrophoneRecorder;
}

/// Method channel implementation for microphone recording.
class _MethodChannelMicrophoneRecorder implements MicrophoneRecorder {
  static const _methodChannel = MethodChannel('qwen_asr/microphone');
  static const _eventChannel = EventChannel('qwen_asr/microphone_stream');

  StreamSubscription<dynamic>? _audioSubscription;
  final _audioController = StreamController<Float32List>.broadcast();

  RecorderState _state = RecorderState.initialized;

  @override
  Stream<Float32List> get audioStream => _audioController.stream;

  @override
  RecorderState get state => _state;

  @override
  bool get isRecording => _state == RecorderState.recording;

  @override
  Future<void> start({AudioFormat format = AudioFormat.qwenAsr}) async {
    if (_state == RecorderState.recording) {
      throw StateError('Already recording');
    }

    if (_audioController.isClosed) {
      throw StateError('Recorder has been disposed');
    }

    try {
      // Start native recording
      await _methodChannel.invokeMethod('start', {
        'sampleRate': format.sampleRate,
        'channels': format.channels,
        'bitsPerSample': format.bitsPerSample,
      });

      // Listen to audio stream
      _audioSubscription = _eventChannel
          .receiveBroadcastStream()
          .listen(
            _onAudioData,
            onError: _onAudioError,
          );

      _state = RecorderState.recording;
    } on PlatformException catch (e) {
      _state = RecorderState.error;
      rethrow;
    }
  }

  void _onAudioData(dynamic data) {
    if (data is List) {
      // Convert List<dynamic> to Float32List
      final samples = Float32List.fromList(
        data.cast<double>().toList(),
      );
      _audioController.add(samples);
    } else if (data is Uint8List) {
      // Raw bytes - convert to float32
      final buffer = data.buffer.asFloat32List();
      _audioController.add(buffer);
    }
  }

  void _onAudioError(dynamic error) {
    _state = RecorderState.error;
    _audioController.addError(error);
  }

  @override
  Future<void> stop() async {
    if (_state != RecorderState.recording) {
      return;
    }

    try {
      await _methodChannel.invokeMethod('stop');
      await _audioSubscription?.cancel();
      _audioSubscription = null;
      _state = RecorderState.stopped;
    } on PlatformException catch (e) {
      _state = RecorderState.error;
      rethrow;
    }
  }

  @override
  Future<void> dispose() async {
    if (isRecording) {
      await stop();
    }
    await _audioSubscription?.cancel();
    if (!_audioController.isClosed) {
      await _audioController.close();
    }
  }
}
