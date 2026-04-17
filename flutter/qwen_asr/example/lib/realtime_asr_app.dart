/// Real-time ASR Demo App with Model Download
///
/// Features:
/// - Download Qwen 0.6B model on first run
/// - Real-time microphone transcription
/// - Visual waveform display
/// - Export transcript to file
///
/// Run: flutter run -t lib/realtime_asr_app.dart
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:permission_handler/permission_handler.dart';
import 'package:qwen_asr/qwen_asr.dart';

void main() {
  runApp(const RealtimeAsrApp());
}

class RealtimeAsrApp extends StatelessWidget {
  const RealtimeAsrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qwen ASR Real-time',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.light,
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
      ),
      home: const RealtimeAsrPage(),
    );
  }
}

class RealtimeAsrPage extends StatefulWidget {
  const RealtimeAsrPage({super.key});

  @override
  State<RealtimeAsrPage> createState() => _RealtimeAsrPageState();
}

class _RealtimeAsrPageState extends State<RealtimeAsrPage> {
  // Model state
  bool _isModelReady = false;
  bool _isDownloading = false;
  double _downloadProgress = 0;
  String _modelStatus = 'Model not loaded';
  QAsrEngine? _engine;

  // Recording state
  bool _isRecording = false;
  MicrophoneRecorder? _recorder;
  QAsrStream? _stream;
  StreamSubscription<Float32List>? _audioSub;
  StreamSubscription<String>? _deltaSub;
  StreamSubscription<String>? _textSub;

  // Transcription state
  String _transcript = '';
  String _currentDelta = '';
  double _recordingSeconds = 0;
  final List<double> _audioLevels = [];
  Timer? _levelTimer;

  // History
  final List<String> _transcriptHistory = [];

  @override
  void initState() {
    super.initState();
    _checkModelAndDownload();
  }

  @override
  void dispose() {
    _cleanup();
    super.dispose();
  }

  void _cleanup() {
    _stopRecording();
    _audioSub?.cancel();
    _deltaSub?.cancel();
    _textSub?.cancel();
    _recorder?.dispose();
    _stream?.dispose();
    _engine?.dispose();
    _levelTimer?.cancel();
  }

  // ==================== Model Management ====================

  Future<void> _checkModelAndDownload() async {
    setState(() => _modelStatus = 'Checking model...');

    // Check if model exists locally
    final modelDir = await _getModelDirectory();
    if (await _isValidModelDirectory(modelDir)) {
      await _loadModel(modelDir.path);
      return;
    }

    // Need to download
    setState(() {
      _isDownloading = true;
      _modelStatus = 'Downloading model (~1.2GB)...';
    });

    await _downloadModel();
  }

  Future<void> _downloadModel() async {
    try {
      final modelDir = await _getModelDirectory();
      await modelDir.create(recursive: true);

      // Hugging Face model repository
      const baseUrl = 'https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main';
      final files = [
        'model.safetensors',
        'vocab.json',
        'config.json',
        'generation_config.json',
        'merges.txt',
        'tokenizer_config.json',
      ];

      for (var i = 0; i < files.length; i++) {
        final file = files[i];
        final url = '$baseUrl/$file';
        final localPath = path.join(modelDir.path, file);

        setState(() {
          _downloadProgress = i / files.length;
          _modelStatus = 'Downloading $file (${(i + 1)}/${files.length})...';
        });

        await _downloadFile(url, localPath);
      }

      setState(() {
        _isDownloading = false;
        _downloadProgress = 1.0;
        _modelStatus = 'Model downloaded';
      });

      await _loadModel(modelDir.path);
    } catch (e) {
      setState(() {
        _isDownloading = false;
        _modelStatus = 'Download failed: $e';
      });
    }
  }

  Future<void> _downloadFile(String url, String localPath) async {
    final file = File(localPath);
    
    // Check if file already exists and has content
    if (await file.exists() && await file.length() > 0) {
      print('File already exists: $localPath');
      return;
    }

    print('Downloading: $url');
    
    final request = http.Request('GET', Uri.parse(url));
    final response = await http.Client().send(request);
    
    if (response.statusCode != 200) {
      throw Exception('Failed to download $url: HTTP ${response.statusCode}');
    }

    final contentLength = response.contentLength ?? 0;
    final sink = file.openWrite();
    var received = 0;
    
    await for (final chunk in response.stream) {
      sink.add(chunk);
      received += chunk.length;
      
      // Update progress for large files
      if (contentLength > 0) {
        final progress = received / contentLength;
        print('Download progress: ${(progress * 100).toStringAsFixed(1)}%');
      }
    }
    
    await sink.close();
    print('Downloaded: $localPath (${received ~/ 1024} KB)');
  }

  Future<Directory> _getModelDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    return Directory(path.join(appDir.path, 'qwen3-asr-0.6b'));
  }

  Future<bool> _isValidModelDirectory(Directory dir) async {
    if (!await dir.exists()) return false;

    final safetensors = File(path.join(dir.path, 'model.safetensors'));
    final vocab = File(path.join(dir.path, 'vocab.json'));

    return await safetensors.exists() && await vocab.exists();
  }

  Future<void> _loadModel(String modelPath) async {
    setState(() => _modelStatus = 'Loading model (this may take 30-60s)...');

    try {
      _engine?.dispose();
      
      // Load model asynchronously (note: this still runs on main thread
      // but allows UI to update. For true background loading, we'd need
      // to refactor the Rust library to support isolate spawning)
      final engine = await QAsrEngine.load(
        modelPath,
        threads: 4,
        verbosity: 1,
      );

      setState(() {
        _engine = engine;
        _isModelReady = true;
        _modelStatus = 'Model ready: ${engine.modelInfo.variant}';
      });
    } on QAsrException catch (e) {
      setState(() => _modelStatus = 'Load failed: $e');
    } catch (e) {
      setState(() => _modelStatus = 'Load error: $e');
    }
  }

  // ==================== Recording ====================

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (_engine == null) return;

    // Request microphone permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      _showError('Microphone permission denied');
      return;
    }

    try {
      // Create stream with low latency config
      _stream = await QAsrStream.create(
        _engine!,
        config: StreamingConfig.lowLatency,
      );

      // Listen to transcription
      _deltaSub = _stream!.deltaStream.listen((delta) {
        setState(() => _currentDelta = delta);
      });

      _textSub = _stream!.textStream.listen((text) {
        setState(() => _transcript = text);
      });

      // Create and start microphone recorder
      _recorder = MicrophoneRecorder.create();

      _audioSub = _recorder!.audioStream.listen((samples) {
        _stream?.addAudio(samples);
        _updateAudioLevel(samples);
      });

      await _recorder!.start();

      // Start recording timer
      _recordingSeconds = 0;
      _levelTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
        setState(() => _recordingSeconds += 0.1);
      });

      setState(() => _isRecording = true);
    } catch (e) {
      _showError('Failed to start recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    setState(() => _isRecording = false);

    // Stop recording
    await _recorder?.stop();
    _levelTimer?.cancel();

    // Finalize stream
    final result = await _stream?.finalize() ?? '';

    // Save to history
    if (result.isNotEmpty) {
      _transcriptHistory.add(result);
    }

    setState(() {
      _transcript = result;
      _currentDelta = '';
    });

    // Cleanup
    await _audioSub?.cancel();
    await _deltaSub?.cancel();
    await _textSub?.cancel();
    _recorder?.dispose();
    _recorder = null;
    _stream?.dispose();
    _stream = null;
  }

  void _updateAudioLevel(Float32List samples) {
    // Calculate RMS level
    double sum = 0;
    for (final sample in samples) {
      sum += sample * sample;
    }
    final rms = (sum / samples.length).sqrt;

    setState(() {
      _audioLevels.add(rms);
      if (_audioLevels.length > 50) {
        _audioLevels.removeAt(0);
      }
    });
  }

  // ==================== UI ====================

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  Future<void> _exportTranscript() async {
    if (_transcript.isEmpty) return;

    try {
      final dir = await getDownloadsDirectory() ?? await getTemporaryDirectory();
      final file = File(
        path.join(dir.path, 'transcript_${DateTime.now().millisecondsSinceEpoch}.txt'),
      );
      await file.writeAsString(_transcript);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Saved to ${file.path}')),
      );
    } catch (e) {
      _showError('Export failed: $e');
    }
  }

  void _clearTranscript() {
    setState(() => _transcript = '');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Real-time ASR'),
        actions: [
          if (_transcript.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.save),
              onPressed: _exportTranscript,
              tooltip: 'Export transcript',
            ),
          if (_transcript.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.clear),
              onPressed: _clearTranscript,
              tooltip: 'Clear transcript',
            ),
        ],
      ),
      body: Column(
        children: [
          // Model status bar
          _buildModelStatusBar(),

          // Audio waveform
          if (_isRecording) _buildWaveform(),

          // Recording controls
          _buildRecordingControls(),

          // Transcription display
          Expanded(child: _buildTranscriptView()),

          // History
          if (_transcriptHistory.isNotEmpty) _buildHistoryView(),
        ],
      ),
    );
  }

  Widget _buildModelStatusBar() {
    Color statusColor;
    if (_isModelReady) {
      statusColor = Colors.green;
    } else if (_isDownloading) {
      statusColor = Colors.orange;
    } else {
      statusColor = Colors.red;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: statusColor.withOpacity(0.1),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                _isModelReady ? Icons.check_circle : Icons.warning,
                color: statusColor,
                size: 16,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  _modelStatus,
                  style: TextStyle(
                    color: statusColor,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          if (_isDownloading) ...[
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _downloadProgress),
          ],
        ],
      ),
    );
  }

  Widget _buildWaveform() {
    return Container(
      height: 60,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: CustomPaint(
        size: const Size(double.infinity, 60),
        painter: WaveformPainter(levels: _audioLevels),
      ),
    );
  }

  Widget _buildRecordingControls() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Recording button
          GestureDetector(
            onTap: _isModelReady ? _toggleRecording : null,
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isRecording ? Colors.red : Colors.deepPurple,
                boxShadow: [
                  BoxShadow(
                    color: (_isRecording ? Colors.red : Colors.deepPurple)
                        .withOpacity(0.3),
                    blurRadius: 20,
                    spreadRadius: 5,
                  ),
                ],
              ),
              child: Icon(
                _isRecording ? Icons.stop : Icons.mic,
                color: Colors.white,
                size: 40,
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Recording timer
          if (_isRecording)
            Text(
              _formatDuration(_recordingSeconds),
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),

          // Current delta
          if (_currentDelta.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                '$_currentDelta▌',
                style: TextStyle(
                  color: Colors.deepPurple.shade300,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildTranscriptView() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: _transcript.isEmpty
          ? const Center(
              child: Text(
                'Tap the microphone to start\nrecording audio...',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey),
              ),
            )
          : SingleChildScrollView(
              child: SelectableText(
                _transcript,
                style: const TextStyle(
                  fontSize: 18,
                  height: 1.5,
                ),
              ),
            ),
    );
  }

  Widget _buildHistoryView() {
    return Container(
      height: 100,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'History',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _transcriptHistory.length,
              itemBuilder: (context, index) {
                return Card(
                  child: Container(
                    width: 200,
                    padding: const EdgeInsets.all(8),
                    child: Text(
                      _transcriptHistory[index],
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  String _formatDuration(double seconds) {
    final mins = (seconds ~/ 60).toString().padLeft(2, '0');
    final secs = (seconds % 60).toInt().toString().padLeft(2, '0');
    return '$mins:$secs';
  }
}

// ==================== Custom Painters ====================

class WaveformPainter extends CustomPainter {
  final List<double> levels;

  WaveformPainter({required this.levels});

  @override
  void paint(Canvas canvas, Size size) {
    if (levels.isEmpty) return;

    final paint = Paint()
      ..color = Colors.deepPurple
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;

    final barWidth = size.width / 50;
    final gap = barWidth * 0.2;

    for (var i = 0; i < levels.length; i++) {
      final level = levels[i].clamp(0.0, 1.0);
      final barHeight = level * size.height;
      final x = i * (barWidth + gap);
      final y = (size.height - barHeight) / 2;

      canvas.drawLine(
        Offset(x + barWidth / 2, y),
        Offset(x + barWidth / 2, y + barHeight),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

// Extension for sqrt
extension on double {
  double get sqrt => this <= 0 ? 0 : _sqrt(this);
}

double _sqrt(double x) {
  var guess = x;
  for (var i = 0; i < 10; i++) {
    guess = (guess + x / guess) / 2;
  }
  return guess;
}
