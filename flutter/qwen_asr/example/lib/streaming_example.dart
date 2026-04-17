/// Example demonstrating real-time streaming transcription.
///
/// Run with: flutter run -t lib/streaming_example.dart
library;

import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qwen_asr/qwen_asr.dart';

void main() {
  runApp(const StreamingDemoApp());
}

class StreamingDemoApp extends StatelessWidget {
  const StreamingDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qwen ASR Streaming Demo',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: const StreamingDemoPage(),
    );
  }
}

class StreamingDemoPage extends StatefulWidget {
  const StreamingDemoPage({super.key});

  @override
  State<StreamingDemoPage> createState() => _StreamingDemoPageState();
}

class _StreamingDemoPageState extends State<StreamingDemoPage> {
  final _modelDirController = TextEditingController();
  
  QAsrEngine? _engine;
  QAsrStream? _stream;
  
  String _status = 'Not loaded';
  String _transcript = '';
  String _currentDelta = '';
  bool _isRecording = false;
  double _processedSeconds = 0;
  
  StreamSubscription<String>? _deltaSub;
  StreamSubscription<String>? _textSub;
  
  @override
  void dispose() {
    _stream?.dispose();
    _engine?.dispose();
    _modelDirController.dispose();
    _deltaSub?.cancel();
    _textSub?.cancel();
    super.dispose();
  }
  
  Future<void> _loadModel() async {
    final dir = _modelDirController.text.trim();
    if (dir.isEmpty) {
      setState(() => _status = 'Please enter model directory');
      return;
    }

    setState(() => _status = 'Loading model...');
    
    try {
      _engine?.dispose();
      final engine = await QAsrEngine.load(dir, verbosity: 1);
      
      setState(() {
        _engine = engine;
        _status = 'Model loaded: ${engine.modelInfo.variant} ${engine.modelInfo.modelType}';
      });
    } on QAsrException catch (e) {
      setState(() => _status = 'Load failed: $e');
    } catch (e) {
      setState(() => _status = 'Unexpected error: $e');
    }
  }
  
  Future<void> _startStreaming() async {
    if (_engine == null) return;
    
    // Create stream with low latency config
    _stream = await QAsrStream.create(
      _engine!,
      config: StreamingConfig.lowLatency,
    );
    
    // Listen to deltas (new words as they arrive)
    _deltaSub = _stream!.deltaStream.listen((delta) {
      setState(() {
        _currentDelta = delta;
        _processedSeconds = _stream!.processedDurationSeconds;
      });
    });
    
    // Listen to full text
    _textSub = _stream!.textStream.listen((text) {
      setState(() {
        _transcript = text;
        _processedSeconds = _stream!.processedDurationSeconds;
      });
    });
    
    setState(() {
      _isRecording = true;
      _status = 'Streaming...';
      _transcript = '';
      _currentDelta = '';
    });
    
    // Simulate feeding audio from file
    // In real app, this would come from microphone
    await _simulateAudioFeed();
  }
  
  Future<void> _simulateAudioFeed() async {
    // Load test audio and feed it in chunks
    try {
      final data = await rootBundle.load('test_fixtures/audio.wav');
      // Parse WAV header (44 bytes) and get PCM data
      final bytes = data.buffer.asUint8List();
      final samples = _parseWav(bytes);
      
      if (samples == null) {
        setState(() => _status = 'Failed to parse test audio');
        return;
      }
      
      // Feed in 0.5 second chunks
      const chunkSize = 8000; // 0.5s at 16kHz
      for (var i = 0; i < samples.length && _isRecording; i += chunkSize) {
        final end = (i + chunkSize).clamp(0, samples.length);
        final chunk = samples.sublist(i, end);
        await _stream!.addAudio(Float32List.fromList(chunk));
        await Future.delayed(const Duration(milliseconds: 100)); // Simulate real-time
      }
      
      if (_isRecording) {
        await _stopStreaming();
      }
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }
  
  List<double>? _parseWav(Uint8List bytes) {
    // Simple WAV parser - assumes 16-bit PCM
    if (bytes.length < 44) return null;
    
    // Check RIFF header
    if (bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46) {
      return null;
    }
    
    // Get format info
    final numChannels = bytes[22] | (bytes[23] << 8);
    final sampleRate = bytes[24] | (bytes[25] << 8) | (bytes[26] << 16) | (bytes[27] << 24);
    final bitsPerSample = bytes[34] | (bytes[35] << 8);
    
    // Find data chunk
    var dataOffset = 44;
    while (dataOffset < bytes.length - 8) {
      if (bytes[dataOffset] == 0x64 && bytes[dataOffset + 1] == 0x61 && 
          bytes[dataOffset + 2] == 0x74 && bytes[dataOffset + 3] == 0x61) {
        break;
      }
      dataOffset++;
    }
    dataOffset += 8; // Skip 'data' and size
    
    // Convert to f32
    final samples = <double>[];
    for (var i = dataOffset; i < bytes.length - 1; i += 2 * numChannels) {
      final sample = bytes[i] | (bytes[i + 1] << 8);
      final signed = sample > 32767 ? sample - 65536 : sample;
      samples.add(signed / 32768.0);
    }
    
    // Resample if needed (simple decimation)
    if (sampleRate == 16000) {
      return samples;
    } else if (sampleRate == 48000) {
      return [
        for (var i = 0; i < samples.length; i++)
          if (i % 3 == 0) samples[i],
      ];
    }
    return samples;
  }
  
  Future<void> _stopStreaming() async {
    setState(() => _isRecording = false);
    
    final result = await _stream?.finalize() ?? '';
    
    setState(() {
      _transcript = result;
      _currentDelta = '';
      _status = 'Streaming complete';
    });
    
    await _deltaSub?.cancel();
    await _textSub?.cancel();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Qwen ASR Streaming Demo'),
        actions: [
          if (_engine != null)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Center(
                child: Text(
                  '${_engine!.modelInfo.variant}',
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _modelDirController,
              decoration: const InputDecoration(
                labelText: 'Model directory path',
                hintText: '/path/to/qwen3-asr-0.6b',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.folder),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _engine == null ? _loadModel : null,
                  icon: const Icon(Icons.download),
                  label: const Text('Load Model'),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: _engine != null && !_isRecording ? _startStreaming : null,
                  icon: const Icon(Icons.mic),
                  label: const Text('Start'),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: _isRecording ? _stopStreaming : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isRecording ? Colors.red : null,
                    foregroundColor: _isRecording ? Colors.white : null,
                  ),
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Status: $_status', style: const TextStyle(fontWeight: FontWeight.bold)),
                  if (_isRecording || _processedSeconds > 0)
                    Text('Processed: ${_processedSeconds.toStringAsFixed(1)}s'),
                ],
              ),
            ),
            const SizedBox(height: 16),
            if (_currentDelta.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  'Current: $_currentDelta',
                  style: TextStyle(color: Colors.blue.shade800),
                ),
              ),
            const SizedBox(height: 8),
            const Text('Transcript:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SingleChildScrollView(
                  child: SelectableText(
                    _transcript.isEmpty ? '(waiting for audio...)' : _transcript,
                    style: const TextStyle(fontSize: 16, height: 1.5),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
