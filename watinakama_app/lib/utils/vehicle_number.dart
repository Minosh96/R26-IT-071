import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

/// Reads the text in a photo on the phone (Google ML Kit, works offline) and
/// returns the first VIN, chassis or engine number it finds, or null if the
/// photo has none. Used to stop photos of random objects before they are sent
/// for VIN analysis.
Future<String?> findVehicleNumberInImage(String imagePath) async {
  final recognizer = TextRecognizer(script: TextRecognitionScript.latin);
  try {
    final result = await recognizer.processImage(InputImage.fromFilePath(imagePath));
    return findVehicleNumber(result.text);
  } catch (_) {
    return null;
  } finally {
    await recognizer.close();
  }
}

// 17-character VIN: letters and digits, never I, O or Q.
final RegExp _vin = RegExp(r'[A-HJ-NPR-Z0-9]{17}');
// Japanese-style chassis or engine number, e.g. HA36S-123456, NZE141-1234567, R06A-1234567.
final RegExp _chassis = RegExp(r'[A-Z0-9]{2,7}-[0-9]{4,8}');
// Other stamped numbers: 8 to 17 letters and digits containing both.
final RegExp _stamped = RegExp(r'(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]{8,17}');

/// Finds a VIN, chassis or engine number in OCR text.
String? findVehicleNumber(String text) {
  for (final rawLine in text.toUpperCase().split('\n')) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;

    // Stamped VINs are often read with stray spaces between characters.
    final compact = line.replaceAll(RegExp(r'\s+'), '');
    final vin = _vin.firstMatch(compact)?.group(0);
    if (vin != null && RegExp(r'[0-9]').hasMatch(vin) && RegExp(r'[A-Z]').hasMatch(vin)) {
      return vin;
    }

    final chassis = _chassis.firstMatch(line.replaceAll(RegExp(r'\s*-\s*'), '-'))?.group(0);
    if (chassis != null && RegExp(r'[A-Z]').hasMatch(chassis)) return chassis;

    for (final word in line.split(RegExp(r'[^A-Z0-9]+'))) {
      final match = _stamped.matchAsPrefix(word);
      if (match != null && match.group(0) == word) return word;
    }
  }
  return null;
}
