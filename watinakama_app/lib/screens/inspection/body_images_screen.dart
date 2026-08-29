import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../widgets/image_sub_stepper.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';
import '../../widgets/loading_overlay.dart';
import '../../widgets/custom_toast.dart';
import '../../widgets/result_sheet.dart';
import '../../widgets/nav_button_row.dart';

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
    // TODO: re-enable required-angles validation once testing is done.
    final images = _capturedPaths.where((p) => p != null).map((p) => File(p!)).toList();

    if (images.isEmpty) {
      Navigator.pushNamed(
        context,
        '/inspection/vin',
        arguments: {
          ..._vehicleData,
          'body_score': null,
          'body_images': _capturedPaths,
          'body_damages': const [],
        },
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    final result = await _apiService.analyzeBody(images);

    if (!mounted) return;
    setState(() {
      _bodyResult = result;
      _isAnalyzing = false;
    });

    if (result['status'] == 'error' || result['status_code'] == 503) {
      String msg = result['message'] ?? 'Body analysis service is unavailable.';
      if (result['status_code'] == 503) msg = "Body models are still loading or missing. Please wait a moment.";
      ToastService.show(context, msg, isError: true);
      return;
    } else {
      _showBodyResult(result);
    }
  }

  void _showBodyResult(Map<String, dynamic> result) {
    final double score = (result['body_condition_score'] ?? result['body_score'] ?? result['score'] ?? 0).toDouble();
    final damages = (result['detected_damage_types'] ?? result['detections'] ?? result['damages'] ?? []) as List;


    final color = AppColors.statusColorFor(score);

    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '${score.toInt()}',
            style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold, color: color),
          ),
          const Text(
            'Body Condition Score',
            style: TextStyle(color: AppColors.textGray, fontSize: 14),
          ),
          const SizedBox(height: 20),
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: score / 100,
              backgroundColor: Colors.black12,
              color: color,
              minHeight: 10,
            ),
          ),
          const SizedBox(height: 24),
          if (damages.isNotEmpty) ...[
            const Align(
              alignment: Alignment.centerLeft,
              child: Text('Detected Issues:', style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 12),
            ...damages.map((d) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    children: [
                      const Icon(Icons.warning_amber_rounded, color: AppColors.statusRed, size: 18),
                      const SizedBox(width: 10),
                      Text(d.toString(), style: const TextStyle(color: AppColors.textGray)),
                    ],
                  ),
                )),
          ] else
            const Text('No significant body damage detected.', style: TextStyle(color: AppColors.statusGreen)),
        ],
      ),
      ctaLabel: 'Continue to VIN Scan',
      onCta: () {
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
    );
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
        message: 'Analyzing body condition...',
        child: Column(
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
              style: const TextStyle(color: AppColors.textWhite, fontSize: 16, fontWeight: FontWeight.bold),
            ),
            
            const SizedBox(height: 16),
            
            // Preview Area
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: AppColors.darkNavySurface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.textFieldBorder),
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
              child: NavButtonRow(
                onBack: () => Navigator.pop(context),
                onNext: _handleNext,
              ),
            ),
          ],
        ),
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
              color: isCamera ? AppColors.primaryBlue : AppColors.darkNavySurface,
              border: isCamera ? Border.all(
                color: Colors.white.withValues(alpha: 0.3),
                width: 3,
              ) : null,
            ),
            child: Icon(
              icon,
              color: isCamera ? Colors.white : AppColors.textWhite,
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
