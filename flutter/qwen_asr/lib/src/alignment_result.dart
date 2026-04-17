import 'dart:typed_data';

/// A single word or character with its time span.
class WordTimestamp {
  final String text;
  final Duration start;
  final Duration end;

  const WordTimestamp({
    required this.text,
    required this.start,
    required this.end,
  });

  /// Duration of this word.
  Duration get duration => end - start;

  @override
  String toString() =>
      'WordTimestamp("$text", ${start.inMilliseconds}ms-${end.inMilliseconds}ms)';
}

/// Result of a forced alignment operation.
class AlignmentResult {
  final String language;
  final List<WordTimestamp> words;

  const AlignmentResult({
    required this.language,
    required this.words,
  });

  /// Total duration of the aligned audio.
  Duration get totalDuration =>
      words.isEmpty ? Duration.zero : words.last.end;

  /// Get words within a time range.
  List<WordTimestamp> wordsInRange(Duration start, Duration end) {
    return words
        .where((w) => w.start >= start && w.end <= end)
        .toList();
  }

  /// Merge words into phrases of roughly equal duration.
  ///
  /// [targetDuration] is the target duration per phrase.
  List<PhraseTimestamp> toPhrases(Duration targetDuration) {
    final phrases = <PhraseTimestamp>[];
    var currentWords = <WordTimestamp>[];
    Duration currentStart = Duration.zero;

    for (final word in words) {
      if (currentWords.isEmpty) {
        currentStart = word.start;
        currentWords.add(word);
      } else if (word.end - currentStart >= targetDuration) {
        // Finish current phrase
        phrases.add(PhraseTimestamp(
          text: currentWords.map((w) => w.text).join(' '),
          start: currentStart,
          end: currentWords.last.end,
          words: List.unmodifiable(currentWords),
        ));
        // Start new phrase
        currentStart = word.start;
        currentWords = [word];
      } else {
        currentWords.add(word);
      }
    }

    // Don't forget the last phrase
    if (currentWords.isNotEmpty) {
      phrases.add(PhraseTimestamp(
        text: currentWords.map((w) => w.text).join(' '),
        start: currentStart,
        end: currentWords.last.end,
        words: List.unmodifiable(currentWords),
      ));
    }

    return phrases;
  }

  /// Export to SRT subtitle format.
  ///
  /// [maxWordsPerLine] controls how many words per subtitle entry.
  String toSrt({int maxWordsPerLine = 8}) {
    return exportSrt(
      result: this,
      maxWordsPerLine: maxWordsPerLine,
    );
  }

  /// Export to WebVTT subtitle format.
  ///
  /// [maxWordsPerLine] controls how many words per subtitle entry.
  String toWebVtt({int maxWordsPerLine = 8}) {
    return exportWebvtt(
      result: this,
      maxWordsPerLine: maxWordsPerLine,
    );
  }
}

/// A phrase composed of multiple words.
class PhraseTimestamp {
  final String text;
  final Duration start;
  final Duration end;
  final List<WordTimestamp> words;

  const PhraseTimestamp({
    required this.text,
    required this.start,
    required this.end,
    required this.words,
  });

  Duration get duration => end - start;
}

/// Export alignment results to SRT subtitle format.
String exportSrt({
  required AlignmentResult result,
  int maxWordsPerLine = 8,
}) {
  final buffer = StringBuffer();
  var entryNum = 1;

  for (var i = 0; i < result.words.length; i += maxWordsPerLine) {
    final chunk = result.words.skip(i).take(maxWordsPerLine).toList();
    if (chunk.isEmpty) continue;

    final start = chunk.first.start;
    final end = chunk.last.end;
    final text = chunk.map((w) => w.text).join(' ');

    buffer.writeln(entryNum);
    buffer.writeln('${_formatSrtTime(start)} --> ${_formatSrtTime(end)}');
    buffer.writeln(text.trim());
    buffer.writeln();

    entryNum++;
  }

  return buffer.toString();
}

/// Export alignment results to WebVTT subtitle format.
String exportWebvtt({
  required AlignmentResult result,
  int maxWordsPerLine = 8,
}) {
  final buffer = StringBuffer();
  buffer.writeln('WEBVTT');
  buffer.writeln();

  for (var i = 0; i < result.words.length; i += maxWordsPerLine) {
    final chunk = result.words.skip(i).take(maxWordsPerLine).toList();
    if (chunk.isEmpty) continue;

    final start = chunk.first.start;
    final end = chunk.last.end;
    final text = chunk.map((w) => w.text).join(' ');

    buffer.writeln('${_formatVttTime(start)} --> ${_formatVttTime(end)}');
    buffer.writeln(text.trim());
    buffer.writeln();
  }

  return buffer.toString();
}

String _formatSrtTime(Duration d) {
  final hours = d.inHours;
  final minutes = d.inMinutes.remainder(60);
  final seconds = d.inSeconds.remainder(60);
  final millis = d.inMilliseconds.remainder(1000);
  return '${_pad2(hours)}:${_pad2(minutes)}:${_pad2(seconds)},${_pad3(millis)}';
}

String _formatVttTime(Duration d) {
  final hours = d.inHours;
  final minutes = d.inMinutes.remainder(60);
  final seconds = d.inSeconds.remainder(60);
  final millis = d.inMilliseconds.remainder(1000);
  return '${_pad2(hours)}:${_pad2(minutes)}:${_pad2(seconds)}.${_pad3(millis)}';
}

String _pad2(int n) => n.toString().padLeft(2, '0');
String _pad3(int n) => n.toString().padLeft(3, '0');
