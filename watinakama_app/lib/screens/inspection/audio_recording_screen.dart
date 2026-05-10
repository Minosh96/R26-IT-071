import 'dart:async';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../widgets/custom_toast.dart';
import '../../services/api_service.dart';
import '../../widgets/loading_overlay.dart';

class AudioRecordingScreen extends StatefulWidget {
  const AudioRecordingScreen({super.key});

  @override
  State<AudioRecordingScreen> createState() => _AudioRecordingScreenState();
}

class _AudioRecordingScreenState extends State<AudioRecordingScreen> with SingleTickerProviderStateMixin {
  final AuthService _auth = AuthService();
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  
  Map<String, dynamic> _vehicleData = {};
  final Map<String, String> _recordingPaths = {};
  final Map<String, Duration> _recordingDurations = {};
  String _userName = '';
  Duration _recordingDuration = Duration.zero;
  Timer? _timer;
  bool _isRecording = false;
  bool _isPaused = false;
  bool _isPlaying = false;
  String _recordingPath = '';
  String? _profilePicPath;
  String _activePhaseKey = 'engine_start';
  bool _isAnalyzing = false;
  Map<String, dynamic>? _engineResult;
  final ApiService _apiService = ApiService();
  String? _audioFilePath;

  // Recording phases
  Map<String, String> _phaseStatus = {
    'engine_start': 'pending',
    'idle': 'pending',
    'acceleration': 'pending',
  };

  late AnimationController _waveController;

  final List<Map<String, dynamic>> _phases = [
    {'key': 'engine_start', 'title': 'Engine Start', 'subtitle': 'Cranking & Cold Start', 'duration': 15},
    {'key': 'idle', 'title': 'Idle Running', 'subtitle': 'Steady State Listening', 'duration': 30},
    {'key': 'acceleration', 'title': 'Acceleration', 'subtitle': 'Light Throttle Press', 'duration': 10},
  ];

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

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String threeDigits(int n) => n.toString().padLeft(3, "0");
    
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    final milliseconds = threeDigits(duration.inMilliseconds.remainder(1000)).substring(0, 2);
    
    return "$minutes:$seconds:$milliseconds";
  }

  Future<void> _startRecording() async {
    try {
      if (await _recorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        _audioFilePath = '${dir.path}/engine_recording.wav';
        
        await _recorder.start(
          const RecordConfig(encoder: AudioEncoder.wav),
          path: _audioFilePath!,
        );
        
        setState(() {
          _isRecording = true;
          _isPaused = false;
          _recordingPath = _audioFilePath!;
          _recordingDuration = Duration.zero;
          _phaseStatus[_activePhaseKey] = 'recording';
        });

        _startTimer();
      }
    } catch (e) {
      ToastService.show(context, "Error starting recorder", isError: true);
    }
  }

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      setState(() {
        _recordingDuration += const Duration(milliseconds: 100);
        _updatePhases();
      });
    });
  }

  void _updatePhases() {
    int totalSeconds = _recordingDuration.inSeconds;
    int maxDuration = _phases.firstWhere((p) => p['key'] == _activePhaseKey)['duration'];
    
    // Update only the active phase
    if (totalSeconds < maxDuration) {
      _phaseStatus[_activePhaseKey] = 'recording';
    } else {
      _phaseStatus[_activePhaseKey] = 'done';
      _stopRecording();
    }
  }

  void _switchPhase(String phaseKey) {
    if (_isRecording || _isPaused) {
      ToastService.show(context, "Please stop recording before switching phases", isError: true);
      return;
    }

    // Save current before switching if it exists
    if (_recordingPath.isNotEmpty) {
      _recordingPaths[_activePhaseKey] = _recordingPath;
      _recordingDurations[_activePhaseKey] = _recordingDuration;
      _phaseStatus[_activePhaseKey] = 'done';
    }

    setState(() {
      _activePhaseKey = phaseKey;
      _recordingPath = _recordingPaths[phaseKey] ?? '';
      _recordingDuration = _recordingDurations[phaseKey] ?? Duration.zero;
      _isPlaying = false;
      if (_player.playing) _player.stop();
    });
  }

  Future<void> _playRecording() async {
    if (_recordingPath.isEmpty) return;
    try {
      if (_isPlaying) {
        await _player.stop();
        setState(() => _isPlaying = false);
      } else {
        await _player.setFilePath(_recordingPath);
        setState(() => _isPlaying = true);
        _player.play().then((_) {
          if (mounted) setState(() => _isPlaying = false);
        });
      }
    } catch (e) {
      ToastService.show(context, "Error playing audio", isError: true);
    }
  }

  Future<void> _pauseRecording() async {
    await _recorder.pause();
    _timer?.cancel();
    setState(() {
      _isRecording = false;
      _isPaused = true;
    });
  }

  Future<void> _resumeRecording() async {
    await _recorder.resume();
    setState(() {
      _isRecording = true;
      _isPaused = false;
    });
    _timer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      setState(() {
        _recordingDuration += const Duration(milliseconds: 100);
        _updatePhases();
      });
    });
  }

  Future<void> _stopRecording() async {
    await _recorder.stop();
    _timer?.cancel();
    setState(() {
      _isRecording = false;
      _isPaused = false;
      _phaseStatus[_activePhaseKey] = 'done';
    });
  }

  void _resetRecording() {
    if (_isRecording || _isPaused) {
      _recorder.stop();
      _timer?.cancel();
    }
    setState(() {
      _isRecording = false;
      _isPaused = false;
      _recordingDuration = Duration.zero;
      _recordingPath = '';
      _isPlaying = false;
      _phaseStatus[_activePhaseKey] = 'pending';
      _recordingPaths.remove(_activePhaseKey);
      _recordingDurations.remove(_activePhaseKey);
    });
  }

  Future<void> _handleNext() async {
    if (_audioFilePath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please record engine audio first')),
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    final result = await _apiService.analyzeEngine(File(_audioFilePath!));

    setState(() {
      _engineResult = result;
      _isAnalyzing = false;
    });

    if (result['status'] == 'success') {
      _showEngineResult(result);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Analysis failed: ${result["message"]}'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _showEngineResult(Map<String, dynamic> result) {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1A2035),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // MHS Score circle
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _getMHSColor(result['mhs_score']).withOpacity(0.2),
                border: Border.all(
                  color: _getMHSColor(result['mhs_score']),
                  width: 3,
                ),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text('${result["mhs_score"]}',
                      style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: _getMHSColor(result['mhs_score']))),
                  const Text('MHS',
                      style: TextStyle(color: Colors.grey, fontSize: 12)),
                ],
              ),
            ),
            const SizedBox(height: 16),
            // Fault class badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: _getMHSColor(result['mhs_score']).withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                (result['fault_class'] as String)
                    .toUpperCase()
                    .replaceAll('_', ' '),
                style: TextStyle(
                  color: _getMHSColor(result['mhs_score']),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 12),
            Text(result['explanation'] ?? '',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.grey, fontSize: 13)),
            const SizedBox(height: 8),
            Text('Confidence: ${result["confidence_percent"]}',
                style: const TextStyle(color: Colors.white70, fontSize: 12)),
            const SizedBox(height: 24),
            // Continue button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF2E7D32),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(24)),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
                onPressed: () {
                  Navigator.pop(context); // close bottom sheet
                  Navigator.pushNamed(
                    context,
                    '/inspection/images',
                    arguments: {
                      ..._vehicleData,
                      'fault_class': result['fault_class'],
                      'confidence': result['confidence'],
                      'mhs_score': result['mhs_score'],
                      'audio_file': _audioFilePath,
                    },
                  );
                },
                child: const Text('Continue to Body Scan',
                    style: TextStyle(
                        color: Colors.white, fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getMHSColor(dynamic score) {
    final s = (score as num).toInt();
    if (s >= 80) return const Color(0xFF00E676);
    if (s >= 50) return const Color(0xFFFFB300);
    return const Color(0xFFFF1744);
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
                    // Recording Card
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1A2035),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Column(
                        children: [
                          Container(
                            width: 48,
                            height: 48,
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: AppColors.darkNavyBg,
                            ),
                            child: Icon(
                              Icons.mic,
                              color: _isRecording ? AppColors.statusRed : Colors.white,
                              size: 28,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _formatDuration(_recordingDuration),
                            style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                              fontFamily: 'monospace',
                            ),
                          ),
                          const SizedBox(height: 12),
                          
                          // Waveform
                          Container(
                            height: 60,
                            width: double.infinity,
                            decoration: BoxDecoration(
                              color: const Color(0xFF0D1117),
                              borderRadius: BorderRadius.circular(30),
                            ),
                            child: _isRecording 
                              ? AnimatedBuilder(
                                  animation: _waveController,
                                  builder: (context, child) {
                                    return CustomPaint(
                                      painter: WaveformPainter(animationValue: _waveController.value),
                                    );
                                  },
                                )
                              : _recordingPath.isNotEmpty
                                  ? Center(
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.center,
                                        children: [
                                          IconButton(
                                            icon: Icon(_isPlaying ? Icons.stop : Icons.play_arrow, color: Colors.white),
                                            onPressed: _playRecording,
                                          ),
                                          const Text("Listen to recording", style: TextStyle(color: Colors.white, fontSize: 14)),
                                        ],
                                      ),
                                    )
                                  : const Center(
                                      child: Text(
                                        "Tap Start to record",
                                        style: TextStyle(color: AppColors.textGray, fontSize: 12),
                                      ),
                                    ),
                          ),
                          const SizedBox(height: 12),
                          
                          // Controls
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            children: [
                              OutlinedButton(
                                onPressed: _resetRecording,
                                style: OutlinedButton.styleFrom(
                                  side: BorderSide(color: _recordingPath.isNotEmpty ? AppColors.statusRed : AppColors.textGray),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                ),
                                child: Row(
                                  children: [
                                    Icon(_recordingPath.isNotEmpty ? Icons.delete_outline : Icons.refresh, 
                                        size: 14, color: _recordingPath.isNotEmpty ? AppColors.statusRed : AppColors.textGray),
                                    const SizedBox(width: 4),
                                    Text(_recordingPath.isNotEmpty ? "Discard" : "Reset", 
                                        style: TextStyle(color: _recordingPath.isNotEmpty ? AppColors.statusRed : AppColors.textGray)),
                                  ],
                                ),
                              ),
                              
                              if (!_isRecording && !_isPaused)
                                ElevatedButton(
                                  onPressed: _recordingPath.isNotEmpty ? _handleNext : _startRecording,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: _recordingPath.isNotEmpty ? const Color(0xFF2E7D32) : AppColors.primaryBlue,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
                                  ),
                                  child: Text(
                                    _recordingPath.isNotEmpty ? "Confirm & Next" : "Start", 
                                    style: const TextStyle(color: Colors.white)
                                  ),
                                ),
                                
                              if (_isRecording)
                                OutlinedButton(
                                  onPressed: _pauseRecording,
                                  style: OutlinedButton.styleFrom(
                                    side: const BorderSide(color: Colors.white),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                  ),
                                  child: const Row(
                                    children: [
                                      Icon(Icons.pause, size: 14, color: Colors.white),
                                      SizedBox(width: 4),
                                      Text("Pause", style: TextStyle(color: Colors.white)),
                                    ],
                                  ),
                                ),
                                
                              if (_isPaused) ...[
                                ElevatedButton(
                                  onPressed: _resumeRecording,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.primaryBlue,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                  ),
                                  child: const Text("Resume", style: TextStyle(color: Colors.white)),
                                ),
                                ElevatedButton(
                                  onPressed: _stopRecording,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: const Color(0xFFB71C1C),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                  ),
                                  child: Row(
                                    children: [
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                                      ),
                                      const SizedBox(width: 6),
                                      const Text("Stop", style: TextStyle(color: Colors.white)),
                                    ],
                                  ),
                                ),
                              ] else if (_isRecording)
                                ElevatedButton(
                                  onPressed: _stopRecording,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: const Color(0xFFB71C1C),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                  ),
                                  child: Row(
                                    children: [
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                                      ),
                                      const SizedBox(width: 6),
                                      const Text("Stop", style: TextStyle(color: Colors.white)),
                                    ],
                                  ),
                                ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    
                    const SizedBox(height: 16),
                    
                    // Phases
                    ...List.generate(_phases.length, (index) {
                      final phase = _phases[index];
                      final status = _phaseStatus[phase['key']];
                      return GestureDetector(
                        onTap: () => _switchPhase(phase['key']),
                        child: Container(
                          margin: const EdgeInsets.only(bottom: 8),
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1A2035),
                            borderRadius: BorderRadius.circular(10),
                            border: _activePhaseKey == phase['key'] 
                                ? Border.all(color: AppColors.primaryBlue, width: 1) 
                                : null,
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    RichText(
                                      text: TextSpan(
                                        children: [
                                          TextSpan(
                                            text: phase['title'],
                                            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
                                          ),
                                          TextSpan(
                                            text: " - ${phase['subtitle']}",
                                            style: const TextStyle(color: AppColors.textGray, fontSize: 13),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              if (status == 'done')
                                const Text("Done", style: TextStyle(color: AppColors.statusGreen, fontSize: 13, fontWeight: FontWeight.bold))
                              else if (status == 'recording')
                                Text(
                                  "${_getRemainingSeconds(phase['key'])} sec",
                                  style: const TextStyle(color: AppColors.statusAmber, fontSize: 13),
                                ),
                            ],
                          ),
                        ),
                      );
                    }),
                    
                    const SizedBox(height: 16),
                    
                    // Tips
                    Row(
                      children: [
                        _buildTipCard(Icons.menu_open, "Bonnet open", "for best pickup"),
                        const SizedBox(width: 8),
                        _buildTipCard(Icons.social_distance, "10 cm away", "from the engine"),
                        const SizedBox(width: 8),
                        _buildTipCard(Icons.back_hand, "Steady hand", "reduce vibration"),
                      ],
                    ),
                    
                    const SizedBox(height: 24),
                    
                    // Bottom Nav
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        ElevatedButton(
                          onPressed: () => Navigator.pop(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF1A1A2E),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                          ),
                          child: const Text("Back", style: TextStyle(color: Colors.white)),
                        ),
                        ElevatedButton(
                          onPressed: _handleNext,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2E7D32),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                          ),
                          child: const Text("Next", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                        ),
                      ],
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

  int _getRemainingSeconds(String phaseKey) {
    Duration duration = (phaseKey == _activePhaseKey) ? _recordingDuration : (_recordingDurations[phaseKey] ?? Duration.zero);
    int current = duration.inSeconds;
    
    int maxDuration = _phases.firstWhere((p) => p['key'] == phaseKey)['duration'];
    
    return (maxDuration - current).clamp(0, maxDuration);
  }

  Widget _buildTipCard(IconData icon, String line1, String line2) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFF1A2035),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Icon(icon, color: AppColors.textGray, size: 22),
            const SizedBox(height: 6),
            Text(line1, style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
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
        paint.color = Colors.white.withOpacity(0.6);
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
