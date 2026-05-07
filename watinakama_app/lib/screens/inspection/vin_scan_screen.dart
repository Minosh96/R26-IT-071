import 'dart:io';
import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:image_picker/image_picker.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';

class VinScanScreen extends StatefulWidget {
  const VinScanScreen({super.key});

  @override
  State<VinScanScreen> createState() => _VinScanScreenState();
}

class _VinScanScreenState extends State<VinScanScreen> {
  Map<String, dynamic> _vehicleData = {};
  List<String?> _bodyImages = [];
  String _userName = '';
  bool _isAnalyzing = false;
  bool _flashOn = false;
  String? _scannedValue;
  String? _capturedImagePath;
  String? _profilePicPath;
  final MobileScannerController _scannerController = MobileScannerController();
  final AuthService _authService = AuthService();

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  @override
  void dispose() {
    _scannerController.dispose();
    super.dispose();
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
          _bodyImages = List<String?>.from(args['body_images'] ?? []);
        });
      }
    });
  }

  void _toggleFlash() {
    _scannerController.toggleTorch();
    setState(() {
      _flashOn = !_flashOn;
    });
  }

  Future<void> _captureFromGallery() async {
    final picker = ImagePicker();
    final photo = await picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      setState(() {
        _capturedImagePath = photo.path;
        _scannedValue = "MANUAL_UPLOAD";
      });
    }
  }

  Future<void> _handleAnalyze() async {
    if (_capturedImagePath == null && _scannedValue == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please scan or upload VIN/Engine number image")),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
    });

    // Simulate analysis delay
    await Future.delayed(const Duration(seconds: 2));

    if (!mounted) return;

    Navigator.pushNamed(
      context,
      '/inspection/results',
      arguments: {
        ..._vehicleData,
        'body_images': _bodyImages,
        'vin_image': _capturedImagePath,
        'vin_status': _scannedValue != null ? 'original' : 'unknown',
        'body_score': 80.0, // placeholder
        'fault_class': _vehicleData['fault_class'] ?? 'healthy',
        'confidence': _vehicleData['confidence'] ?? 1.0,
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
      body: Column(
        children: [
          Container(
            height: 75,
            width: double.infinity,
            color: AppColors.lightBlueTop,
            alignment: Alignment.center,
            child: const ProgressStepper(currentStep: 3),
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Identify Vehicle",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  RichText(
                    text: const TextSpan(
                      style: TextStyle(fontSize: 13, color: AppColors.textGray),
                      children: [
                        TextSpan(text: "Position the "),
                        TextSpan(
                          text: "VIN",
                          style: TextStyle(
                            color: AppColors.primaryBlue,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        TextSpan(text: " or "),
                        TextSpan(
                          text: "Engine Number",
                          style: TextStyle(
                            color: Color(0xFFFF8C00),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        TextSpan(
                          text: " within the frame to scan automatically.",
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  // Scanner frame
                  Container(
                    height: 200,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: const Color(0xFF0D1117),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Stack(
                      children: [
                        if (_capturedImagePath != null)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: Image.file(
                              File(_capturedImagePath!),
                              fit: BoxFit.cover,
                              width: double.infinity,
                              height: double.infinity,
                            ),
                          )
                        else
                          ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: MobileScanner(
                              controller: _scannerController,
                              onDetect: (capture) {
                                final List<Barcode> barcodes = capture.barcodes;
                                if (barcodes.isNotEmpty && barcodes.first.rawValue != null) {
                                  setState(() {
                                    _scannedValue = barcodes.first.rawValue;
                                  });
                                  _scannerController.stop();
                                }
                              },
                            ),
                          ),
                        // Corner brackets overlay
                        CustomPaint(
                          painter: ScannerOverlayPainter(),
                          child: Container(),
                        ),
                        // Silhouette
                        Center(
                          child: Icon(
                            Icons.directions_car_outlined,
                            color: Colors.white.withOpacity(0.15),
                            size: 80,
                          ),
                        ),
                        // Scanning indicator
                        Positioned(
                          bottom: 12,
                          left: 0,
                          right: 0,
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                width: 8,
                                height: 8,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: _scannedValue == null
                                      ? AppColors.statusGreen
                                      : AppColors.statusGreen,
                                ),
                              ),
                              const SizedBox(width: 6),
                              Text(
                                _scannedValue == null ? "SCANNING..." : "DETECTED",
                                style: TextStyle(
                                  color: _scannedValue == null
                                      ? Colors.white
                                      : AppColors.statusGreen,
                                  fontSize: 12,
                                  letterSpacing: 1.5,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  // Tip card
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2A2000),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFF4A3800)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          width: 32,
                          height: 32,
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            color: Color(0xFF4A3800),
                          ),
                          child: const Center(
                            child: Icon(
                              Icons.lightbulb_outline,
                              color: Color(0xFFFFB300),
                              size: 18,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                "Better Lighting",
                                style: TextStyle(
                                  color: Color(0xFFFFB300),
                                  fontSize: 13,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              SizedBox(height: 4),
                              Text(
                                "Ensure the vehicle numbers are clearly visible and well-lit",
                                style: TextStyle(
                                  color: Color(0xFFCCA000),
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                  // Action buttons
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildActionButton(
                        icon: Icons.flash_on,
                        label: "Flash",
                        onTap: _toggleFlash,
                        isActive: _flashOn,
                      ),
                      GestureDetector(
                        onTap: () {
                          // Trigger analyze if something is detected/captured
                          if (_scannedValue != null || _capturedImagePath != null) {
                            _handleAnalyze();
                          }
                        },
                        child: Container(
                          width: 64,
                          height: 64,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: AppColors.primaryBlue,
                            border: Border.all(
                              color: Colors.white.withOpacity(0.3),
                              width: 3,
                            ),
                          ),
                          child: const Icon(Icons.camera_alt, color: Colors.white, size: 30),
                        ),
                      ),
                      _buildActionButton(
                        icon: Icons.photo_library_outlined,
                        label: "Gallery",
                        onTap: _captureFromGallery,
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  // Navigation buttons
                  Row(
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
                          padding: const EdgeInsets.symmetric(
                            horizontal: 28,
                            vertical: 14,
                          ),
                        ),
                        child: const Text("Back"),
                      ),
                      ElevatedButton(
                        onPressed: _isAnalyzing ? null : _handleAnalyze,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF2E7D32),
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: const Color(0xFF2E7D32).withOpacity(0.6),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(24),
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 14,
                          ),
                        ),
                        child: _isAnalyzing
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                ),
                              )
                            : const Text("Analyze"),
                      ),
                    ],
                  ),
                ],
              ),
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
    bool isActive = false,
  }) {
    return Column(
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 52,
            height: 52,
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              color: Color(0xFF1A2035),
            ),
            child: Icon(
              icon,
              color: isActive ? const Color(0xFFFFB300) : Colors.white,
              size: 24,
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

class ScannerOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.primaryBlue
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    const double length = 20;
    const double inset = 16;

    // Top Left
    canvas.drawLine(const Offset(inset, inset), const Offset(inset + length, inset), paint);
    canvas.drawLine(const Offset(inset, inset), const Offset(inset, inset + length), paint);

    // Top Right
    canvas.drawLine(
        Offset(size.width - inset, inset), Offset(size.width - inset - length, inset), paint);
    canvas.drawLine(
        Offset(size.width - inset, inset), Offset(size.width - inset, inset + length), paint);

    // Bottom Left
    canvas.drawLine(
        Offset(inset, size.height - inset), Offset(inset + length, size.height - inset), paint);
    canvas.drawLine(
        Offset(inset, size.height - inset), Offset(inset, size.height - inset - length), paint);

    // Bottom Right
    canvas.drawLine(Offset(size.width - inset, size.height - inset),
        Offset(size.width - inset - length, size.height - inset), paint);
    canvas.drawLine(Offset(size.width - inset, size.height - inset),
        Offset(size.width - inset, size.height - inset - length), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
