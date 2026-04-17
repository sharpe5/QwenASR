/// On-device Qwen3-ASR speech-to-text for Flutter.
///
/// ## Quick Start
///
/// ```dart
/// import 'package:qwen_asr/qwen_asr.dart';
///
/// // Load model
/// final engine = await QAsrEngine.load('/path/to/model');
///
/// // Batch transcription
/// final text = await engine.transcribeFile('audio.wav');
///
/// // Real-time streaming
/// final stream = await QAsrStream.create(engine);
/// stream.textStream.listen(print);
/// await stream.finalize();
///
/// // Forced alignment (word timestamps)
/// final alignment = await engine.alignWords(samples, transcript);
/// print(alignment.toSrt());
///
/// // Real-time with microphone
/// final recorder = MicrophoneRecorder.create();
/// recorder.audioStream.listen((chunk) => stream.addAudio(chunk));
/// await recorder.start();
/// ```
library;

// Core engine
export 'src/qwen_asr_engine.dart';

// Streaming
export 'src/qwen_asr_stream.dart';

// Forced alignment (exports WordTimestamp, AlignmentResult)
export 'src/alignment_result.dart';

// Model management (exports ModelInfo)
export 'src/model_manager.dart';

// Audio / Microphone
export 'src/audio/microphone_recorder.dart';

// Generated types - only export types not already exported above
export 'src/rust/api/streaming.dart' show StreamConfig;
