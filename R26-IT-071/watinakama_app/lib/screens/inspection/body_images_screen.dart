import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../widgets/image_sub_stepper.dart';
import '../../services/auth_service.dart';

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

  void _handleNext() {
    if (!_allCaptured) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please capture all 5 angles before continuing")),
      );
    } else {
      Navigator.pushNamed(
        context,
        '/inspection/vin',
        arguments: {
          ..._vehicleData,
          'body_images': _capturedPaths,
        },
      );
    }
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
      body: Column(
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
          
          // Actions
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    ElevatedButton(
                      onPressed: () => _pickImage(ImageSource.camera),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primaryBlue,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                      ),
                      child: Text(_capturedPaths[_currentAngle] == null ? "Capture" : "Retake"),
                    ),
                    const SizedBox(width: 8),
                    IconButton(
                      icon: const Icon(Icons.photo_library_outlined, color: Colors.white70),
                      onPressed: () => _pickImage(ImageSource.gallery),
                    ),
                  ],
                ),
                ElevatedButton(
                  onPressed: _handleNext,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _allCaptured ? const Color(0xFF2E7D32) : Colors.grey.shade800,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                  ),
                  child: const Text("Next", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
