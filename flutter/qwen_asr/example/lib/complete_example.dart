/// Complete example demonstrating all Qwen ASR features.
///
/// Features shown:
/// - Model loading with error handling
/// - Batch transcription
/// - Real-time streaming
/// - Forced alignment (word timestamps)
/// - Model asset bundling
/// - SRT/WebVTT export
///
/// Run: flutter run -t lib/complete_example.dart
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:qwen_asr/qwen_asr.dart';

void main() {
  runApp(const CompleteDemoApp());
}

class CompleteDemoApp extends StatelessWidget {
  const CompleteDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qwen ASR Complete Demo',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const CompleteDemoPage(),
    );
  }
}

class CompleteDemoPage extends StatefulWidget {
  const CompleteDemoPage({super.key});

  @override
  State<CompleteDemoPage> createState() => _CompleteDemoPageState();
}

class _CompleteDemoPageState extends State<CompleteDemoPage>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  // Shared state
  QAsrEngine? _engine;
  String _modelPath = '';

  // Batch transcription state
  String _batchResult = '';
  String _batchStatus = 'Ready';

  // Streaming state
  QAsrStream? _stream;
  String _streamResult = '';
  String _streamDelta = '';
  bool _isStreaming = false;
  double _streamDuration = 0;

  // Alignment state
  String _alignmentStatus = 'Ready';
  List<WordTimestamp> _alignedWords = [];
  String _subtitleOutput = '';

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _stream?.dispose();
    _engine?.dispose();
    super.dispose();
  }

  // ==================== Model Loading ====================

  Future<void> _loadModel() async {
    // Try to use bundled model first
    String modelPath;
    try {
      modelPath = await ModelManager.extractModelFromAssets(
        'assets/models/qwen3-asr-0.6b',
        modelName: 'qwen3-asr-0.6b',
      );
    } catch (e) {
      // Fall back to manual path entry
      modelPath = _modelPath;
      if (modelPath.isEmpty) {
        _showError('Please enter model path or add to assets');
        return;
      }
    }

    try {
      setState(() => _batchStatus = 'Loading model...');

      _engine?.dispose();
      final engine = await QAsrEngine.load(
        modelPath,
        verbosity: 1,
      );

      setState(() {
        _engine = engine;
        final info = engine.modelInfo;
        _batchStatus = 'Loaded: ${info.variant} ${info.modelType}';
      });
    } on QAsrException catch (e) {
      _showError('Failed to load model: $e');
    }
  }

  // ==================== Batch Transcription ====================

  Future<void> _transcribeFile() async {
    if (_engine == null) {
      _showError('Load a model first');
      return;
    }

    setState(() {
      _batchStatus = 'Transcribing...';
      _batchResult = '';
    });

    try {
      // For demo, load from assets
      final data = await rootBundle.load('test_fixtures/audio.wav');
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/test_audio.wav');
      await tempFile.writeAsBytes(data.buffer.asUint8List());

      final result = await _engine!.transcribeFile(tempFile.path);

      setState(() {
        _batchResult = result;
        _batchStatus = 'Done! ${_engine!.perfStats()}';
      });

      await tempFile.delete();
    } catch (e) {
      _showError('Transcription failed: $e');
    }
  }

  // ==================== Streaming ====================

  Future<void> _startStreaming() async {
    if (_engine == null) {
      _showError('Load a model first');
      return;
    }

    setState(() {
      _isStreaming = true;
      _streamResult = '';
      _streamDelta = '';
      _streamDuration = 0;
    });

    // Create stream with low latency
    _stream = await QAsrStream.create(
      _engine!,
      config: const StreamingConfig(
        chunkSec: 0.5,
        rollbackTokens: 3,
        unfixedChunks: 1,
      ),
    );

    // Listen to results
    _stream!.deltaStream.listen((delta) {
      setState(() {
        _streamDelta = delta;
        _streamDuration = _stream!.processedDurationSeconds;
      });
    });

    _stream!.textStream.listen((text) {
      setState(() {
        _streamResult = text;
        _streamDuration = _stream!.processedDurationSeconds;
      });
    });

    // Simulate audio feed from file
    await _simulateAudioFeed();
  }

  Future<void> _simulateAudioFeed() async {
    try {
      final data = await rootBundle.load('test_fixtures/audio.wav');
      final samples = _parseWavPcm(data.buffer.asUint8List());

      if (samples == null) {
        _showError('Failed to parse audio');
        return;
      }

      // Feed in chunks
      const chunkSize = 8000; // 0.5s
      for (var i = 0;
          i < samples.length && _isStreaming;
          i += chunkSize) {
        final end = (i + chunkSize).clamp(0, samples.length);
        final chunk = samples.sublist(i, end);
        await _stream!.addAudio(Float32List.fromList(chunk));
        await Future.delayed(const Duration(milliseconds: 200));
      }

      if (_isStreaming) {
        await _stopStreaming();
      }
    } catch (e) {
      _showError('Streaming error: $e');
    }
  }

  Future<void> _stopStreaming() async {
    setState(() => _isStreaming = false);
    final result = await _stream?.finalize() ?? '';
    setState(() => _streamResult = result);
  }

  // ==================== Forced Alignment ====================

  Future<void> _alignTranscript() async {
    if (_engine == null) {
      _showError('Load a model first');
      return;
    }

    if (!_engine!.isAligner) {
      _showError('Model is not an aligner. Use qwen3-aligner-0.6b');
      return;
    }

    if (_batchResult.isEmpty) {
      _showError('Transcribe audio first to get a transcript');
      return;
    }

    setState(() => _alignmentStatus = 'Aligning...');

    try {
      // Load audio samples
      final data = await rootBundle.load('test_fixtures/audio.wav');
      final samples = _parseWavPcm(data.buffer.asUint8List());

      if (samples == null) {
        _showError('Failed to parse audio');
        return;
      }

      final result = await _engine!.alignWords(
        Float32List.fromList(samples),
        _batchResult,
        language: 'English',
      );

      setState(() {
        _alignedWords = result.words;
        _alignmentStatus = 'Aligned ${result.words.length} words';
      });

      // Generate subtitles
      _generateSubtitles(result);
    } catch (e) {
      _showError('Alignment failed: $e');
    }
  }

  void _generateSubtitles(AlignmentResult result) {
    // Use phrase-level grouping for better subtitles
    final phrases = result.toPhrases(const Duration(seconds: 3));

    final srt = StringBuffer();
    for (var i = 0; i < phrases.length; i++) {
      final phrase = phrases[i];
      srt.writeln(i + 1);
      srt.writeln(
          '${_formatSrtTime(phrase.start)} --> ${_formatSrtTime(phrase.end)}');
      srt.writeln(phrase.text);
      srt.writeln();
    }

    setState(() => _subtitleOutput = srt.toString());
  }

  // ==================== Helpers ====================

  List<double>? _parseWavPcm(Uint8List bytes) {
    // Simple WAV parser
    if (bytes.length < 44) return null;

    final sampleRate = bytes[24] |
        (bytes[25] << 8) |
        (bytes[26] << 16) |
        (bytes[27] << 24);

    // Find data chunk
    var offset = 36;
    while (offset < bytes.length - 8) {
      if (bytes[offset] == 0x64 &&
          bytes[offset + 1] == 0x61 &&
          bytes[offset + 2] == 0x74 &&
          bytes[offset + 3] == 0x61) {
        break;
      }
      offset++;
    }
    offset += 8;

    // Convert to f32
    final samples = <double>[];
    for (var i = offset; i < bytes.length - 1; i += 2) {
      final sample = bytes[i] | (bytes[i + 1] << 8);
      final signed = sample > 32767 ? sample - 65536 : sample;
      samples.add(signed / 32768.0);
    }

    return samples;
  }

  String _formatSrtTime(Duration d) {
    final h = d.inHours.toString().padLeft(2, '0');
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    final ms = d.inMilliseconds.remainder(1000).toString().padLeft(3, '0');
    return '$h:$m:$s,$ms';
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
    setState(() {
      _batchStatus = 'Error';
      _alignmentStatus = 'Error';
    });
  }

  // ==================== UI ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Qwen ASR Complete Demo'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.file_copy), text: 'Batch'),
            Tab(icon: Icon(Icons.mic), text: 'Streaming'),
            Tab(icon: Icon(Icons.timer), text: 'Alignment'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildBatchTab(),
          _buildStreamingTab(),
          _buildAlignmentTab(),
        ],
      ),
    );
  }

  Widget _buildBatchTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            decoration: const InputDecoration(
              labelText: 'Model path (or use assets)',
              hintText: '/path/to/qwen3-asr-0.6b',
              border: OutlineInputBorder(),
            ),
            onChanged: (v) => _modelPath = v,
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              ElevatedButton(
                onPressed: _loadModel,
                child: const Text('Load Model'),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: _engine != null ? _transcribeFile : null,
                child: const Text('Transcribe File'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('Status: $_batchStatus',
              style: const TextStyle(fontWeight: FontWeight.bold)),
          if (_engine != null)
            Text('Model: ${_engine!.modelInfo.variant} '
                '(${_engine!.modelInfo.modelType})'),
          const SizedBox(height: 16),
          const Text('Result:',
              style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SingleChildScrollView(
                child: SelectableText(
                    _batchResult.isEmpty ? '(empty)' : _batchResult),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStreamingTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              ElevatedButton(
                onPressed:
                    _engine != null && !_isStreaming ? _startStreaming : null,
                child: const Text('Start Streaming'),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: _isStreaming ? _stopStreaming : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isStreaming ? Colors.red : null,
                ),
                child: const Text('Stop'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('Duration: ${_streamDuration.toStringAsFixed(1)}s'),
          if (_streamDelta.isNotEmpty)
            Container(
              padding: const EdgeInsets.all(8),
              color: Colors.blue.shade50,
              child: Text('Delta: $_streamDelta'),
            ),
          const SizedBox(height: 8),
          const Text('Result:',
              style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SingleChildScrollView(
                child: Text(_streamResult.isEmpty ? '(empty)' : _streamResult),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAlignmentTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ElevatedButton(
            onPressed: _alignTranscript,
            child: const Text('Align Transcript'),
          ),
          const SizedBox(height: 12),
          Text('Status: $_alignmentStatus'),
          const SizedBox(height: 12),
          Expanded(
            flex: 2,
            child: _alignedWords.isEmpty
                ? const Center(child: Text('No alignment data'))
                : ListView.builder(
                    itemCount: _alignedWords.length,
                    itemBuilder: (context, index) {
                      final word = _alignedWords[index];
                      return ListTile(
                        dense: true,
                        title: Text(word.text),
                        subtitle: Text(
                            '${word.start.inMilliseconds}ms - ${word.end.inMilliseconds}ms'),
                      );
                    },
                  ),
          ),
          const Divider(),
          const Text('Subtitles (SRT):',
              style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(
            flex: 2,
            child: Container(
              padding: const EdgeInsets.all(8),
              color: Colors.grey.shade100,
              child: SingleChildScrollView(
                child: SelectableText(
                    _subtitleOutput.isEmpty ? '(empty)' : _subtitleOutput),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
