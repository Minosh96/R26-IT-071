import 'dart:async';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:file_picker/file_picker.dart';

import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../widgets/custom_toast.dart';
import '../../services/api_service.dart';
import '../../widgets/loading_overlay.dart';
import '../../widgets/app_card.dart';
import '../../widgets/nav_button_row.dart';
import '../../widgets/result_sheet.dart';
import '../../widgets/score_badge.dart';
import '../../utils/file_validation.dart';

/// One guided recording: the user starts the engine, lets it idle, then revs
/// gently, following on-screen prompts. The server cuts the recording into the
/// same three stages at the times the prompts changed.
class AudioRecordingScreen extends StatefulWidget {
  const AudioRecordingScreen({super.key});

  @override
  State<AudioRecordingScreen> createState() => _AudioRecordingScreenState();
}

class _GuideStep {
  final String title;
  final String instruction;
  final IconData icon;
  final int from; // seconds
  final int to;
  const _GuideStep(this.title, this.instruction, this.icon, this.from, this.to);
}

class _AudioRecordingScreenState extends State<AudioRecordingScreen> with SingleTickerProviderStateMixin {
  // Step boundaries in seconds. They are sent with the upload so the server
  // cuts the recording where the prompts changed.
  static const int _startEnd = 8;
  static const int _idleEnd = 20;
  static const int _totalSeconds = 30;
  static const int _minSeconds = 20; // must match MIN_DURATION_FULL on the server

  static const List<_GuideStep> _steps = [
    _GuideStep('Start the engine', 'Turn the key now', Icons.key, 0, _startEnd),
    _GuideStep('Let it idle', "Don't touch the accelerator", Icons.hourglass_bottom, _startEnd, _idleEnd),
    _GuideStep('Rev gently', 'Press the accelerator a few times', Icons.speed, _idleEnd, _totalSeconds),
  ];

  final AuthService _auth = AuthService();
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final ApiService _apiService = ApiService();

  Map<String, dynamic> _vehicleData = {};
  String _userName = '';
  String? _profilePicPath;

  Duration _elapsed = Duration.zero;
  Timer? _timer;
  bool _isRecording = false;
  bool _isPlaying = false;
  bool _isAnalyzing = false;

  /// The recording or uploaded file to analyse, and whether it was recorded
  /// here with the guided timings (an uploaded file uses the server defaults).
  String _audioPath = '';
  bool _recordedInApp = false;

  /// Error from the server or a local check, shown on the card.
  String? _errorMessage;
  /// Soft warnings from a successful analysis, e.g. no engine start heard.
  List<String> _warnings = [];

  late AnimationController _waveController;

  @override
  void initState() {
    super.initState();
    _waveController = AnimationController(vsync: this, duration: const Duration(milliseconds: 500))..repeat();
    _loadUserName();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;
    if (args is Map<String, dynamic>) {
      _vehicleData = args;
    }
  }

  Future<void> _loadUserName() async {
    final name = await _auth.getUserName();
    final pic = await _auth.getProfilePicPath();
    if (mounted) {
      setState(() {
        _userName = name;
        _profilePicPath = pic;
      });
    }
  }

  int get _elapsedSeconds => _elapsed.inSeconds;

  int get _currentStepIndex {
    for (int i = 0; i < _steps.length; i++) {
      if (_elapsedSeconds < _steps[i].to) return i;
    }
    return _steps.length - 1;
  }

  String _formatDuration(Duration d) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(d.inMinutes.remainder(60))}:${two(d.inSeconds.remainder(60))}';
  }

  Future<void> _startRecording() async {
    try {
      if (!await _recorder.hasPermission()) {
        if (mounted) ToastService.show(context, 'Microphone permission is needed to record', isError: true);
        return;
      }
      final dir = await getApplicationDocumentsDirectory();
      final path = '${dir.path}/engine_recording.wav';
      await _recorder.start(const RecordConfig(encoder: AudioEncoder.wav), path: path);

      setState(() {
        _isRecording = true;
        _elapsed = Duration.zero;
        _audioPath = '';
        _recordedInApp = true;
        _errorMessage = null;
        _warnings = [];
      });

      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 100), (_) {
        setState(() => _elapsed += const Duration(milliseconds: 100));
        if (_elapsedSeconds >= _totalSeconds) _stopRecording();
      });
    } catch (e) {
      if (mounted) ToastService.show(context, 'Error starting recorder', isError: true);
    }
  }

  Future<void> _stopRecording() async {
    _timer?.cancel();
    final path = await _recorder.stop();
    if (!mounted) return;
    setState(() {
      _isRecording = false;
      if (path != null) _audioPath = path;
      if (_elapsedSeconds < _minSeconds) {
        _errorMessage = 'Keep recording for at least $_minSeconds seconds: start, idle, then rev gently.';
      }
    });
  }

  Future<void> _playRecording() async {
    if (_audioPath.isEmpty) return;
    try {
      if (_isPlaying) {
        await _player.stop();
        setState(() => _isPlaying = false);
      } else {
        await _player.setFilePath(_audioPath);
        setState(() => _isPlaying = true);
        _player.play().then((_) {
          if (mounted) setState(() => _isPlaying = false);
        });
      }
    } catch (e) {
      if (mounted) ToastService.show(context, 'Error playing audio', isError: true);
    }
  }

  Future<void> _pickAudioFile() async {
    if (_isRecording) return;
    try {
      final result = await FilePicker.platform.pickFiles(type: FileType.any);
      if (result == null || result.files.single.path == null) return;
      final path = result.files.single.path!;

      final validationError = validateFileExtension(path, allowedAudioExtensions, 'WAV, MP3, M4A, or MP4 audio');
      if (validationError != null) {
        if (mounted) ToastService.show(context, validationError, isError: true);
        return;
      }

      setState(() {
        _audioPath = path;
        _recordedInApp = false;
        _elapsed = Duration.zero;
        _errorMessage = null;
        _warnings = [];
      });
      if (mounted) ToastService.show(context, 'Audio file selected');
    } catch (e) {
      if (mounted) ToastService.show(context, 'Error picking audio file', isError: true);
    }
  }

  void _discardRecording() {
    if (_isRecording) {
      _recorder.stop();
      _timer?.cancel();
    }
    if (_player.playing) _player.stop();
    setState(() {
      _isRecording = false;
      _isPlaying = false;
      _elapsed = Duration.zero;
      _audioPath = '';
      _errorMessage = null;
      _warnings = [];
    });
  }

  Future<void> _handleNext() async {
    if (_isRecording) {
      ToastService.show(context, 'Please wait until the recording finishes', isError: true);
      return;
    }
    if (_audioPath.isEmpty) {
      ToastService.show(context, 'Please record the engine or upload an audio file first.', isError: true);
      return;
    }
    if (_recordedInApp && _elapsedSeconds < _minSeconds) {
      ToastService.show(context, 'The recording is too short. Please record again.', isError: true);
      return;
    }

    setState(() => _isAnalyzing = true);
    final result = await _apiService.analyzeEngineRecording(
      _audioPath,
      startEnd: _recordedInApp ? _startEnd.toDouble() : null,
      idleEnd: _recordedInApp ? _idleEnd.toDouble() : null,
    );
    if (!mounted) return;
    setState(() => _isAnalyzing = false);

    final status = result['status'];
    if (status == 'success') {
      final warnings = <String>[];
      final stages = result['stages'];
      if (stages is Map) {
        for (final stage in stages.values) {
          if (stage is Map && stage['warning'] != null) warnings.add(stage['warning'].toString());
        }
      }
      setState(() {
        _errorMessage = null;
        _warnings = warnings;
      });
      _showEngineResult(result);
    } else {
      final message = result['message']?.toString() ?? 'The recording could not be analysed. Please record again.';
      setState(() => _errorMessage = message);
      ToastService.show(context, message, isError: true);
    }
  }

  void _showEngineResult(Map<String, dynamic> result) {
    final mhsScore = (result['mhs_score'] as num).toDouble();
    final color = AppColors.statusColorFor(mhsScore);

    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ScoreBadge(score: mhsScore, label: 'MHS'),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              (result['fault_class'] as String).toUpperCase().replaceAll('_', ' '),
              style: TextStyle(color: color, fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(height: 12),
          Text(result['explanation'] ?? '',
              textAlign: TextAlign.center, style: const TextStyle(color: AppColors.textGray, fontSize: 13)),
          const SizedBox(height: 8),
          Text('Confidence: ${result["confidence_percent"]}',
              style: const TextStyle(color: AppColors.textGray, fontSize: 12)),
          for (final warning in _warnings) ...[
            const SizedBox(height: 10),
            Text(warning,
                textAlign: TextAlign.center, style: const TextStyle(color: AppColors.statusAmber, fontSize: 12)),
          ],
        ],
      ),
      ctaLabel: 'Continue to Body Scan',
      onCta: () {
        Navigator.pop(context); // close bottom sheet
        Navigator.pushNamed(
          context,
          '/inspection/images',
          arguments: {
            ..._vehicleData,
            'fault_class': result['fault_class'],
            'confidence': result['confidence'],
            'mhs_score': result['mhs_score'],
            'audio_file': _audioPath,
          },
        );
      },
    );
  }

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    _player.dispose();
    _waveController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final step = _steps[_currentStepIndex];
    final hasRecording = _audioPath.isNotEmpty;

    return Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      appBar: InspectionAppBar(
        onBack: () => Navigator.pop(context),
        userName: _userName,
        userPhotoUrl: _profilePicPath,
      ),
      body: LoadingOverlay(
        isLoading: _isAnalyzing,
        message: 'Analyzing engine audio...',
        child: Column(
          children: [
            Container(
              height: 75,
              width: double.infinity,
              color: AppColors.lightBlueTop,
              alignment: Alignment.center,
              child: const ProgressStepper(currentStep: 1),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    AppCard(
                      child: Column(
                        children: [
                          // What to do right now
                          if (_isRecording) ...[
                            Icon(step.icon, color: AppColors.primaryBlue, size: 36),
                            const SizedBox(height: 8),
                            Text(step.title,
                                style: const TextStyle(
                                    color: AppColors.textWhite, fontSize: 22, fontWeight: FontWeight.bold)),
                            const SizedBox(height: 4),
                            Text(step.instruction,
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: AppColors.textGray, fontSize: 14)),
                            const SizedBox(height: 4),
                            Text('${(step.to - _elapsedSeconds).clamp(0, step.to)} s',
                                style: const TextStyle(
                                    color: AppColors.statusAmber, fontSize: 16, fontWeight: FontWeight.bold)),
                          ] else ...[
                            Container(
                              width: 48,
                              height: 48,
                              decoration: const BoxDecoration(shape: BoxShape.circle, color: AppColors.darkNavyBg),
                              child: const Icon(Icons.mic, color: AppColors.textWhite, size: 28),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              hasRecording
                                  ? (_recordedInApp ? 'Recording ready' : 'Audio file selected')
                                  : 'One recording, about $_totalSeconds seconds',
                              style: const TextStyle(
                                  color: AppColors.textWhite, fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                            if (!hasRecording)
                              const Padding(
                                padding: EdgeInsets.only(top: 4),
                                child: Text(
                                  'With the engine off, tap Start and follow the prompts.',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(color: AppColors.textGray, fontSize: 13),
                                ),
                              ),
                          ],
                          const SizedBox(height: 12),

                          // Progress through the 30 seconds
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: (_elapsed.inMilliseconds / (_totalSeconds * 1000)).clamp(0.0, 1.0),
                              minHeight: 6,
                              backgroundColor: AppColors.darkNavySurface,
                              color: AppColors.primaryBlue,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '${_formatDuration(_elapsed)} / ${_formatDuration(const Duration(seconds: _totalSeconds))}',
                            style: const TextStyle(color: AppColors.textGray, fontSize: 12, fontFamily: 'monospace'),
                          ),
                          const SizedBox(height: 12),

                          // Waveform while recording, playback afterwards
                          Container(
                            height: 60,
                            width: double.infinity,
                            decoration: BoxDecoration(
                              color: const Color(0xFF0B0F17),
                              borderRadius: BorderRadius.circular(30),
                            ),
                            child: _isRecording
                                ? AnimatedBuilder(
                                    animation: _waveController,
                                    builder: (context, child) =>
                                        CustomPaint(painter: WaveformPainter(animationValue: _waveController.value)),
                                  )
                                : hasRecording
                                    ? Center(
                                        child: Row(
                                          mainAxisAlignment: MainAxisAlignment.center,
                                          children: [
                                            IconButton(
                                              icon: Icon(_isPlaying ? Icons.stop : Icons.play_arrow,
                                                  color: Colors.white),
                                              onPressed: _playRecording,
                                            ),
                                            const Text('Listen to recording',
                                                style: TextStyle(color: Colors.white, fontSize: 14)),
                                          ],
                                        ),
                                      )
                                    : const Center(
                                        child: Text('Tap Start to record',
                                            style: TextStyle(color: AppColors.textGray, fontSize: 12)),
                                      ),
                          ),

                          if (_errorMessage != null) ...[
                            const SizedBox(height: 12),
                            _messageRow(_errorMessage!, AppColors.statusRed, Icons.error_outline),
                          ],
                          for (final warning in _warnings) ...[
                            const SizedBox(height: 8),
                            _messageRow(warning, AppColors.statusAmber, Icons.warning_amber_rounded),
                          ],
                          const SizedBox(height: 12),

                          // Controls
                          Wrap(
                            alignment: WrapAlignment.center,
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              if (hasRecording || _isRecording)
                                _outlinedButton(Icons.delete_outline, 'Discard', AppColors.statusRed, _discardRecording),
                              if (!_isRecording)
                                _outlinedButton(Icons.upload_file, 'Upload', AppColors.textGray, _pickAudioFile),
                              if (_isRecording)
                                _filledButton('Stop', AppColors.statusRed, _stopRecording)
                              else if (hasRecording && _errorMessage == null)
                                _filledButton('Analyze', AppColors.primaryBlue, _handleNext)
                              else
                                _filledButton(hasRecording ? 'Record again' : 'Start', AppColors.primaryBlue,
                                    _startRecording),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),

                    // The three steps the prompts will walk through
                    ...List.generate(_steps.length, (i) {
                      final s = _steps[i];
                      final active = _isRecording && i == _currentStepIndex;
                      final done = (_isRecording && i < _currentStepIndex) ||
                          (!_isRecording && hasRecording && _recordedInApp && _elapsedSeconds >= s.to);
                      return Container(
                        margin: const EdgeInsets.only(bottom: 8),
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppColors.darkNavySurface,
                          borderRadius: BorderRadius.circular(10),
                          border: active ? Border.all(color: AppColors.primaryBlue, width: 1) : null,
                        ),
                        child: Row(
                          children: [
                            Icon(s.icon, size: 20, color: active ? AppColors.primaryBlue : AppColors.textGray),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text('${i + 1}. ${s.title}',
                                      style: const TextStyle(
                                          color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 14)),
                                  Text(s.instruction, style: const TextStyle(color: AppColors.textGray, fontSize: 12)),
                                ],
                              ),
                            ),
                            Text(
                              done ? 'Done' : '${s.from}-${s.to} s',
                              style: TextStyle(
                                color: done ? AppColors.statusGreen : AppColors.textGray,
                                fontSize: 13,
                                fontWeight: done ? FontWeight.bold : FontWeight.normal,
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                    const SizedBox(height: 16),

                    Row(
                      children: [
                        _buildTipCard(Icons.menu_open, 'Bonnet open', 'for best pickup'),
                        const SizedBox(width: 8),
                        _buildTipCard(Icons.social_distance, '10 cm away', 'from the engine'),
                        const SizedBox(width: 8),
                        _buildTipCard(Icons.volume_off, 'Keep quiet', 'no talking or music'),
                      ],
                    ),
                    const SizedBox(height: 24),

                    NavButtonRow(
                      onBack: () => Navigator.pop(context),
                      onNext: _handleNext,
                    ),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _messageRow(String text, Color color, IconData icon) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 6),
        Expanded(child: Text(text, style: TextStyle(color: color, fontSize: 12))),
      ],
    );
  }

  Widget _outlinedButton(IconData icon, String label, Color color, VoidCallback onPressed) {
    return OutlinedButton(
      onPressed: onPressed,
      style: OutlinedButton.styleFrom(
        side: BorderSide(color: color == AppColors.textGray ? AppColors.textFieldBorder : color),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(color: color, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _filledButton(String label, Color color, VoidCallback onPressed) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      ),
      child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 13)),
    );
  }

  Widget _buildTipCard(IconData icon, String line1, String line2) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: AppColors.darkNavySurface,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Icon(icon, color: AppColors.textGray, size: 22),
            const SizedBox(height: 6),
            Text(line1,
                style: const TextStyle(color: AppColors.textWhite, fontSize: 10, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center),
            Text(line2, style: const TextStyle(color: AppColors.textGray, fontSize: 9), textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

class WaveformPainter extends CustomPainter {
  final double animationValue;
  WaveformPainter({required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.fill
      ..strokeCap = StrokeCap.round;

    int barCount = 40;
    double barWidth = 4;
    double spacing = (size.width - (barCount * barWidth)) / (barCount + 1);

    for (int i = 0; i < barCount; i++) {
      // Random-ish height based on animation and index
      double heightFactor = (0.2 + 0.8 * (0.5 + 0.5 * (i % 5 == 0 ? 0.8 : 0.4))).clamp(0.2, 1.0);
      double height = size.height * heightFactor * (0.6 + 0.4 * (0.5 + 0.5 * (i % 2 == 0 ? 0.5 : -0.5)));

      // Animate height slightly
      height = height * (0.8 + 0.2 * (0.5 + 0.5 * (i % 3 == 0 ? animationValue : 1 - animationValue)));

      double x = spacing + i * (barWidth + spacing);
      double y = (size.height - height) / 2;

      // Red for center bars, white/gray for outer
      if (i > 15 && i < 25) {
        paint.color = AppColors.statusRed;
      } else {
        paint.color = Colors.white.withValues(alpha: 0.6);
      }

      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(x, y, barWidth, height),
          const Radius.circular(2),
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant WaveformPainter oldDelegate) => true;
}
