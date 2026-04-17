import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

/// Manages ASR model files bundled as Flutter assets.
///
/// Models are extracted from assets on first use and cached for
/// subsequent launches.
///
/// ```dart
/// // pubspec.yaml:
/// flutter:
///   assets:
///     - assets/models/qwen3-asr-0.6b/
///
/// // Usage:
/// final modelPath = await ModelManager.extractModelFromAssets(
///   'assets/models/qwen3-asr-0.6b',
///   modelName: 'qwen3-asr-0.6b',
/// );
/// final engine = await QAsrEngine.load(modelPath);
/// ```
class ModelManager {
  static final _extractedModels = <String, String>{};

  /// Extract a model from Flutter assets to the app's documents directory.
  ///
  /// [assetPath] is the path prefix in assets (e.g., 'assets/models/qwen3-asr-0.6b').
  /// [modelName] is a unique name for caching. If null, uses the basename of [assetPath].
  ///
  /// Returns the path to the extracted model directory.
  /// Subsequent calls with the same [modelName] return the cached path.
  static Future<String> extractModelFromAssets(
    String assetPath, {
    String? modelName,
  }) async {
    final name = modelName ?? path.basename(assetPath);

    // Return cached path if already extracted
    if (_extractedModels.containsKey(name)) {
      final cachedPath = _extractedModels[name]!;
      if (await Directory(cachedPath).exists()) {
        return cachedPath;
      }
      // Cache invalidated, remove
      _extractedModels.remove(name);
    }

    // Get or create model directory
    final modelDir = await _getModelDirectory();
    final targetDir = Directory(path.join(modelDir.path, name));

    // Check if already extracted
    if (await targetDir.exists()) {
      // Verify it has required files
      if (await _isValidModelDirectory(targetDir)) {
        _extractedModels[name] = targetDir.path;
        return targetDir.path;
      }
      // Invalid, re-extract
      await targetDir.delete(recursive: true);
    }

    await targetDir.create(recursive: true);

    // Load asset manifest to find all model files
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final allAssets = manifest.listAssets();

    // Find all assets under the given path
    final modelAssets = allAssets
        .where((asset) =>
            asset.startsWith(assetPath) &&
            !asset.endsWith('/')) // Skip directories
        .toList();

    if (modelAssets.isEmpty) {
      throw ModelExtractionException(
        'No model files found at asset path: $assetPath',
      );
    }

    // Extract each file
    for (final asset in modelAssets) {
      final relativePath = asset.substring(assetPath.length);
      final targetPath = path.join(targetDir.path, relativePath);

      // Create parent directories
      final parentDir = Directory(path.dirname(targetPath));
      if (!await parentDir.exists()) {
        await parentDir.create(recursive: true);
      }

      // Copy asset to file
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List();
      await File(targetPath).writeAsBytes(bytes);
    }

    // Verify extraction
    if (!await _isValidModelDirectory(targetDir)) {
      throw ModelExtractionException(
        'Extracted model is missing required files. '
        'Expected model*.safetensors and vocab.json',
      );
    }

    _extractedModels[name] = targetDir.path;
    return targetDir.path;
  }

  /// Get the path to a previously extracted model.
  ///
  /// Returns null if the model hasn't been extracted.
  static Future<String?> getExtractedModelPath(String modelName) async {
    if (_extractedModels.containsKey(modelName)) {
      return _extractedModels[modelName];
    }

    final modelDir = await _getModelDirectory();
    final targetDir = Directory(path.join(modelDir.path, modelName));

    if (await targetDir.exists() && await _isValidModelDirectory(targetDir)) {
      _extractedModels[modelName] = targetDir.path;
      return targetDir.path;
    }

    return null;
  }

  /// Clear all extracted models to free disk space.
  static Future<void> clearCache() async {
    final modelDir = await _getModelDirectory();
    if (await modelDir.exists()) {
      await modelDir.delete(recursive: true);
    }
    _extractedModels.clear();
  }

  /// Clear a specific model from cache.
  static Future<void> clearModel(String modelName) async {
    final modelDir = await _getModelDirectory();
    final targetDir = Directory(path.join(modelDir.path, modelName));

    if (await targetDir.exists()) {
      await targetDir.delete(recursive: true);
    }
    _extractedModels.remove(modelName);
  }

  /// Get the total size of cached models in bytes.
  static Future<int> getCacheSize() async {
    final modelDir = await _getModelDirectory();
    if (!await modelDir.exists()) return 0;

    int totalSize = 0;
    await for (final entity in modelDir.list(recursive: true, followLinks: false)) {
      if (entity is File) {
        totalSize += await entity.length();
      }
    }
    return totalSize;
  }

  /// List all cached models.
  static Future<List<String>> listCachedModels() async {
    final modelDir = await _getModelDirectory();
    if (!await modelDir.exists()) return [];

    final models = <String>[];
    await for (final entity in modelDir.list()) {
      if (entity is Directory) {
        final name = path.basename(entity.path);
        if (await _isValidModelDirectory(entity)) {
          models.add(name);
        }
      }
    }
    return models;
  }

  /// Get the directory where models are stored.
  static Future<Directory> _getModelDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    return Directory(path.join(appDir.path, 'qwen_asr_models'));
  }

  /// Check if a directory contains a valid model.
  static Future<bool> _isValidModelDirectory(Directory dir) async {
    bool hasSafetensors = false;
    bool hasVocab = false;

    await for (final entity in dir.list()) {
      if (entity is File) {
        final name = path.basename(entity.path);
        if (name.startsWith('model') && name.endsWith('.safetensors')) {
          hasSafetensors = true;
        }
        if (name == 'vocab.json') {
          hasVocab = true;
        }
      }
    }

    return hasSafetensors && hasVocab;
  }
}

/// Exception thrown when model extraction fails.
class ModelExtractionException implements Exception {
  final String message;

  const ModelExtractionException(this.message);

  @override
  String toString() => 'ModelExtractionException: $message';
}

/// Information about an available bundled model.
class BundledModelInfo {
  final String name;
  final String variant; // "0.6B" or "1.7B"
  final String type; // "ASR" or "ForcedAligner"
  final bool isAligner;
  final int sizeBytes;

  const BundledModelInfo({
    required this.name,
    required this.variant,
    required this.type,
    required this.isAligner,
    required this.sizeBytes,
  });

  String get sizeFormatted {
    if (sizeBytes >= 1024 * 1024 * 1024) {
      return '${(sizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    } else if (sizeBytes >= 1024 * 1024) {
      return '${(sizeBytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    } else {
      return '${(sizeBytes / 1024).toStringAsFixed(1)} KB';
    }
  }
}

/// Predefined model configurations.
class PredefinedModels {
  /// Qwen3-ASR 0.6B model (~1.2GB).
  static const asr06b = BundledModelInfo(
    name: 'qwen3-asr-0.6b',
    variant: '0.6B',
    type: 'ASR',
    isAligner: false,
    sizeBytes: 1200000000,
  );

  /// Qwen3-ASR 1.7B model (~3.4GB).
  static const asr17b = BundledModelInfo(
    name: 'qwen3-asr-1.7b',
    variant: '1.7B',
    type: 'ASR',
    isAligner: false,
    sizeBytes: 3400000000,
  );

  /// Qwen3-Aligner 0.6B model (~1.2GB).
  static const aligner06b = BundledModelInfo(
    name: 'qwen3-aligner-0.6b',
    variant: '0.6B',
    type: 'ForcedAligner',
    isAligner: true,
    sizeBytes: 1200000000,
  );

  /// All predefined models.
  static const all = [asr06b, asr17b, aligner06b];
}

/// @deprecated Use BundledModelInfo instead
typedef ModelInfo = BundledModelInfo;
