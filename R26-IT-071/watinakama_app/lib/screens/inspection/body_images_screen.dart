import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../widgets/image_sub_stepper.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';

class BodyImagesScreen extends StatefulWidget {
  const BodyImagesScreen({super.key});

  @override
  State<BodyImagesScreen> createState() => _BodyImagesScreenState();
}

class _BodyImagesScreenState extends State<BodyImagesScreen> {
  Map<String, dynamic> _vehicleData = {};
  String _userName = '';
  String? _profilePicPath;
  int _currentAngle = 0; // The angle currently being viewed/captured
  final List<String> _angleNames = ['Front', 'Rear', 'Left', 'Right', 'Up'];
  final List<String?> _capturedPaths = [null, null, null, null, null];
  final AuthService _authService = AuthService();
  bool _isAnalyzing = false;
  Map<String, dynamic>? _bodyResult;
  final ApiService _apiService = ApiService();

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  Future<void> _loadInitialData() async {
    final name = await _authService.getUserName();
    final pic = await _authService.getProfilePicPath();
    setState(() {
      _userName = name;
      _profilePicPath = pic;
    });
    
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final args = ModalRoute.of(context)?.settings.arguments;
      if (args is Map<String, dynamic>) {
        setState(() {
          _vehicleData = args;
        });
      }
    });
  }

  List<bool> get _capturedAngles => _capturedPaths.map((p) => p != null).toList();
  bool get _allCaptured => _capturedPaths.every((p) => p != null);
  String get _currentAngleName => _angleNames[_currentAngle];

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final photo = await picker.pickImage(
      source: source,
      imageQuality: 85,
    );
    
    if (photo != null) {
      setState(() {
        _capturedPaths[_currentAngle] = photo.path;
        // Auto-advance to the next uncaptured angle if we just captured the current one
        if (_allCaptured) return;
        
        for (int i = 0; i < 5; i++) {
          if (_capturedPaths[i] == null) {
            _currentAngle = i;
            break;
          }
        }
      });
    }
  }

  Future<void> _handleNext() async {
    if (!_allCaptured) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please capture all 5 angles')),
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    final images = _capturedPaths.map((p) => File(p!)).toList();
    final result = await _apiService.analyzeBody(images);

    setState(() {
      _bodyResult = result;
      _isAnalyzing = false;
    });

    if (result['status'] == 'error') {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Analysis failed: ${result["message"]}'),
          backgroundColor: Colors.red,
        ),
      );
    } else {
      _showBodyResult(result);
    }
  }

  void _showBodyResult(Map<String, dynamic> result) {
    final double score = (result['body_score'] ?? result['score'] ?? 0).toDouble();
    final damages = (result['detections'] ?? result['damages'] ?? []) as List;

    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1A2035),
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: EdgeInsets.fromLTRB(
            24, 24, 24, MediaQuery.of(context).viewInsets.bottom + 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '${score.toInt()}',
              style: TextStyle(
                fontSize: 48,
                fontWeight: FontWeight.bold,
                color: _getScoreColor(score),
              ),
            ),
            const Text(
              'Body Condition Score',
              style: TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 20),
            // Progress bar
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: LinearProgressIndicator(
                value: score / 100,
                backgroundColor: Colors.white10,
                color: _getScoreColor(score),
                minHeight: 10,
              ),
            ),
            const SizedBox(height: 24),
            if (damages.isNotEmpty) ...[
              const Align(
                alignment: Alignment.centerLeft,
                child: Text('Detected Issues:',
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(height: 12),
              ...damages.map((d) => Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(
                      children: [
                        const Icon(Icons.warning_amber_rounded,
                            color: Colors.redAccent, size: 18),
                        const SizedBox(width: 10),
                        Text(d.toString(),
                            style: const TextStyle(color: Colors.grey)),
                      ],
                    ),
                  )),
            ] else
              const Text('No significant body damage detected.',
                  style: TextStyle(color: AppColors.statusGreen)),
            const SizedBox(height: 32),
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
                    '/inspection/vin',
                    arguments: {
                      ..._vehicleData,
                      'body_score': score,
                      'body_images': _capturedPaths,
                      'body_damages': damages,
                    },
                  );
                },
                child: const Text('Continue to VIN Scan',
                    style: TextStyle(
                        color: Colors.white, fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 80) return const Color(0xFF00E676);
    if (score >= 50) return const Color(0xFFFFB300);
    return const Color(0xFFFF1744);
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
      body: Stack(
        children: [
          Column(
            children: [
              // Step Progress
              Container(
                height: 75,
                width: double.infinity,
                color: AppColors.lightBlueTop,
                alignment: Alignment.center,
                child: const ProgressStepper(currentStep: 2),
              ),
              
              const SizedBox(height: 12),
              
              // Sub-stepper (Tap to switch angle)
              ImageSubStepper(
                currentAngle: _currentAngle,
                capturedAngles: _capturedAngles,
                onAngleSelected: (index) {
                  setState(() => _currentAngle = index);
                },
              ),
              
              const SizedBox(height: 16),
              
              // Header Text
              Text(
                _capturedPaths[_currentAngle] == null 
                    ? "Capture the $_currentAngleName view" 
                    : "$_currentAngleName view captured",
                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
              ),
              
              const SizedBox(height: 16),
              
              // Preview Area
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: const Color(0xFF0D1117),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white10),
                    ),
                    child: _capturedPaths[_currentAngle] != null
                        ? Stack(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(12),
                                child: Image.file(
                                  File(_capturedPaths[_currentAngle]!),
                                  fit: BoxFit.cover,
                                  width: double.infinity,
                                  height: double.infinity,
                                ),
                              ),
                              // Retake indicator
                              Positioned(
                                top: 12,
                                right: 12,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.black54,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: const Text("Tap Capture to Retake", style: TextStyle(color: Colors.white70, fontSize: 10)),
                                ),
                              ),
                            ],
                          )
                        : Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Icon(Icons.camera_alt_outlined, color: AppColors.textGray, size: 48),
                              const SizedBox(height: 12),
                              Text(
                                "Tap Capture for $_currentAngleName view",
                                style: const TextStyle(color: AppColors.textGray, fontSize: 13),
                              ),
                            ],
                          ),
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Action buttons row
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildActionButton(
                      icon: Icons.camera_alt,
                      label: _capturedPaths[_currentAngle] == null ? "Capture" : "Retake",
                      onTap: () => _pickImage(ImageSource.camera),
                      isCamera: true,
                    ),
                    _buildActionButton(
                      icon: Icons.photo_library_outlined,
                      label: "Gallery",
                      onTap: () => _pickImage(ImageSource.gallery),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Navigation buttons
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 30),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    ElevatedButton(
                      onPressed: () => Navigator.pop(context),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1A1A2E),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(24),
                        ),
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                      ),
                      child: const Text("Back"),
                    ),
                    ElevatedButton(
                      onPressed: _handleNext,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF2E7D32),
                        foregroundColor: Colors.white,
                        disabledBackgroundColor: const Color(0xFF2E7D32).withOpacity(0.6),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(24),
                        ),
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                      ),
                      child: const Text(
                        "Next",
                        style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (_isAnalyzing)
            Container(
              color: Colors.black54,
              child: const Center(
                child: CircularProgressIndicator(color: AppColors.primaryBlue),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isCamera = false,
  }) {
    return Column(
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: isCamera ? 64 : 52,
            height: isCamera ? 64 : 52,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isCamera ? AppColors.primaryBlue : const Color(0xFF1A2035),
              border: isCamera ? Border.all(
                color: Colors.white.withOpacity(0.3),
                width: 3,
              ) : null,
            ),
            child: Icon(
              icon,
              color: Colors.white,
              size: isCamera ? 30 : 24,
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          style: const TextStyle(color: AppColors.textGray, fontSize: 11),
        ),
      ],
    );
  }
}
