# Real-time ASR Demo App

This example app demonstrates real-time speech-to-text transcription using the Qwen ASR Flutter plugin with live microphone input.

## Features

- 📥 **Automatic Model Download**: Downloads the Qwen 0.6B model on first run (~1.2GB)
- 🎤 **Real-time Transcription**: Live microphone input with low-latency streaming
- 📊 **Visual Waveform**: Real-time audio level visualization
- 💾 **Export**: Save transcripts to device storage
- 📜 **History**: Keep track of previous recordings

## Prerequisites

### Android
- Android SDK 21+
- Microphone permission (automatically requested)
- Internet permission for model download

### iOS
- iOS 12+
- Xcode 14+
- Microphone permission description in `Info.plist` (already configured)

## Running the App

```bash
# Navigate to example directory
cd flutter/qwen_asr/example

# Get dependencies
flutter pub get

# Run on Android
flutter run -t lib/realtime_asr_app.dart

# Run on iOS (requires macOS)
flutter run -t lib/realtime_asr_app.dart
```

## App Structure

```
lib/
├── realtime_asr_app.dart    # Main app with model download + real-time ASR
├── main.dart                # Original example (batch transcription)
├── streaming_example.dart   # Streaming without microphone
└── complete_example.dart    # All features demo
```

## How It Works

### 1. Model Management
```dart
// Check if model exists locally
final modelDir = await _getModelDirectory();
if (!await _isValidModelDirectory(modelDir)) {
  // Download from your server/HuggingFace
  await _downloadModel();
}

// Load the model
final engine = await QAsrEngine.load(modelDir.path);
```

### 2. Microphone Recording
```dart
// Create recorder
final recorder = MicrophoneRecorder.create();

// Listen to audio chunks
recorder.audioStream.listen((Float32List samples) {
  // Feed to ASR stream
  stream.addAudio(samples);
});

// Start recording
await recorder.start(format: AudioFormat.qwenAsr);
```

### 3. Real-time Transcription
```dart
// Create streaming session
final stream = await QAsrStream.create(
  engine,
  config: StreamingConfig.lowLatency, // Fast response
);

// Listen to results
stream.deltaStream.listen((delta) {
  print('New words: $delta');
});

stream.textStream.listen((text) {
  print('Full transcript: $text');
});
```

## Model Download

The app can automatically download the model from your server or HuggingFace:

```dart
// Update these URLs in _downloadModel()
const baseUrl = 'https://huggingface.co/your-model-repo/resolve/main';
final files = [
  'model.safetensors',
  'vocab.json',
  'config.json',
];
```

For production, you should:
1. Host the model files on your own CDN
2. Implement resume-support for large downloads
3. Add download progress UI
4. Verify file integrity with checksums

## Configuration

### Streaming Config Presets

```dart
// Low latency for real-time captions
StreamingConfig.lowLatency = StreamingConfig(
  chunkSec: 0.5,           // 500ms chunks
  rollbackTokens: 3,       // Minimal rollback
  unfixedChunks: 1,        // Emit immediately
  maxNewTokens: 16,        // Limit generation
);

// High accuracy for offline processing
StreamingConfig.highAccuracy = StreamingConfig(
  chunkSec: 3.0,
  rollbackTokens: 8,
  unfixedChunks: 3,
  maxNewTokens: 64,
  pastTextConditioning: true,
);
```

### Custom Configuration

```dart
final stream = await QAsrStream.create(
  engine,
  config: StreamingConfig(
    chunkSec: 1.0,           // 1 second chunks
    rollbackTokens: 5,       // Token rollback window
    unfixedChunks: 2,        // Wait 2 chunks before emitting
    maxNewTokens: 32,        // Max tokens per chunk
    pastTextConditioning: true, // Use previous context
  ),
);
```

## Permissions

### Android

Already configured in `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

### iOS

Already configured in `ios/Runner/Info.plist`:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs microphone access to transcribe your speech in real-time.</string>
```

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify model URLs are accessible
- Ensure sufficient storage space (~1.5GB)

### Microphone Not Working
- Grant microphone permission in app settings
- Check if another app is using the microphone
- Restart the app

### High Latency
- Use `StreamingConfig.lowLatency`
- Reduce `chunkSec` to 0.5
- Ensure device is not in low-power mode

### Poor Recognition Quality
- Speak clearly and at moderate volume
- Reduce background noise
- Use `StreamingConfig.highAccuracy` for better quality
- Ensure 16kHz sample rate

## Performance Tips

1. **Model Loading**: Load the model once and reuse
2. **Streaming Config**: Use low latency for UI responsiveness
3. **Audio Format**: Stick to 16kHz, mono, float32
4. **Buffer Size**: Larger buffers = smoother but more latency
5. **Thread Count**: Use 4 threads for optimal performance

## Next Steps

- Add VAD (Voice Activity Detection) for auto-stop
- Implement speaker diarization
- Add support for multiple languages
- Create a background recording service
- Add cloud backup for transcripts

## License

Same as the main plugin - see LICENSE file.
