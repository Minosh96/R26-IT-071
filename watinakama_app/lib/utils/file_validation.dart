/// Extensions accepted by the engine-audio analysis backend.
const Set<String> allowedAudioExtensions = {'.wav', '.mp3', '.m4a', '.mp4'};

/// Extensions accepted by the body-condition and VIN analysis backends.
const Set<String> allowedImageExtensions = {
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.heic',
  '.heif',
};

/// Returns a friendly error message if [path]'s extension isn't in [allowed],
/// or null if the file type is acceptable.
String? validateFileExtension(
  String path,
  Set<String> allowed,
  String friendlyTypeList,
) {
  final dotIndex = path.lastIndexOf('.');
  final ext = dotIndex == -1 ? '' : path.substring(dotIndex).toLowerCase();
  if (!allowed.contains(ext)) {
    final shown = ext.isEmpty ? 'unknown' : ext;
    return 'Unsupported file type ($shown). Please select a $friendlyTypeList file.';
  }
  return null;
}
