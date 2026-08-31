import 'dart:io';
import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';
import '../../widgets/loading_overlay.dart';
import '../../widgets/custom_toast.dart';
import '../../widgets/result_sheet.dart';
import '../../widgets/nav_button_row.dart';
import '../../utils/file_validation.dart';

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
  Map<String, dynamic>? _vinResult;
  final ApiService _apiService = ApiService();
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
      final validationError = validateFileExtension(
        photo.path,
        allowedImageExtensions,
        'JPG, PNG, WEBP, or HEIC image',
      );
      if (validationError != null) {
        if (!mounted) return;
        ToastService.show(context, validationError, isError: true);
        return;
      }

      final croppedPath = await _cropImage(photo.path);
      if (croppedPath == null || !mounted) return;
      setState(() {
        _capturedImagePath = croppedPath;
        _scannedValue = "MANUAL_UPLOAD";
      });
    }
  }

  Future<String?> _cropImage(String sourcePath) async {
    final cropped = await ImageCropper().cropImage(
      sourcePath: sourcePath,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop VIN Image',
          toolbarColor: const Color(0xFF0B0F17),
          toolbarWidgetColor: Colors.white,
          backgroundColor: const Color(0xFF0B0F17),
          activeControlsWidgetColor: AppColors.primaryBlue,
          lockAspectRatio: false,
        ),
        IOSUiSettings(
          title: 'Crop VIN Image',
        ),
      ],
    );
    return cropped?.path;
  }

  Future<void> _handleAnalyze() async {
    // TODO: re-enable required-image validation once testing is done.
    if (_capturedImagePath == null) {
      Navigator.pushNamed(
        context,
        '/inspection/results',
        arguments: {
          ..._vehicleData,
          'vin_status': 'unknown',
          'vin_image': null,
          'body_score': _vehicleData['body_score'],
          'mhs_score': _vehicleData['mhs_score'],
          'fault_class': _vehicleData['fault_class'],
        },
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    final result = await _apiService.scanVin(File(_capturedImagePath!));

    if (!mounted) return;
    setState(() {
      _vinResult = result;
      _isAnalyzing = false;
    });

    if (result['status'] == 'error' || result['status_code'] == 503) {
      String msg = result['message'] ?? 'VIN analysis failed.';
      if (result['status_code'] == 503) msg = "VIN models are still loading or missing. Please wait a moment.";
      ToastService.show(context, msg, isError: true);
    } else {
      _showVinResult(result);
    }
  }

  void _showVinResult(Map<String, dynamic> result) {
    final prediction =
        (result['label'] ?? result['prediction'] ?? 'unknown').toString().toLowerCase();
    final confidence = (result['confidence'] ?? 0.0).toDouble();

    IconData icon;
    Color color;
    String statusText;
    String? warning;

    if (prediction == 'original') {
      icon = Icons.check_circle;
      color = AppColors.statusGreen;
      statusText = "VIN ORIGINAL";
    } else if (prediction == 'altered') {
      icon = Icons.warning;
      color = AppColors.statusRed;
      statusText = "VIN ALTERED";
      warning = "Do not purchase this vehicle. The VIN has been tampered with.";
    } else {
      icon = Icons.help_outline;
      color = AppColors.statusAmber;
      statusText = "NEEDS REVIEW";
    }

    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 64),
          const SizedBox(height: 16),
          Text(
            statusText,
            style: TextStyle(color: color, fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
            style: const TextStyle(color: AppColors.textGray, fontSize: 14),
          ),
          if (warning != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                warning,
                textAlign: TextAlign.center,
                style: TextStyle(color: color, fontWeight: FontWeight.w500),
              ),
            ),
          ],
        ],
      ),
      ctaLabel: 'View Full Inspection Report',
      onCta: () {
        Navigator.pop(context); // close bottom sheet
        Navigator.pushNamed(
          context,
          '/inspection/results',
          arguments: {
            ..._vehicleData,
            'vin_status': prediction,
            'vin_image': _capturedImagePath,
            'body_score': _vehicleData['body_score'],
            'mhs_score': _vehicleData['mhs_score'],
            'fault_class': _vehicleData['fault_class'],
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
        message: 'Authenticating VIN...',
        child: Column(
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
                        color: AppColors.textWhite,
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
                        color: Colors.black,
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
                          if (_capturedImagePath != null)
                            Positioned(
                              top: 10,
                              right: 10,
                              child: GestureDetector(
                                onTap: () async {
                                  final croppedPath = await _cropImage(_capturedImagePath!);
                                  if (croppedPath == null || !mounted) return;
                                  setState(() => _capturedImagePath = croppedPath);
                                },
                                child: Container(
                                  padding: const EdgeInsets.all(6),
                                  decoration: BoxDecoration(
                                    color: Colors.black54,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: const Icon(Icons.crop, color: Colors.white, size: 18),
                                ),
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
                              color: Colors.white.withValues(alpha: 0.15),
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
                                  decoration: const BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: AppColors.statusGreen,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  _scannedValue == null ? "SCANNING..." : "DETECTED",
                                  style: TextStyle(
                                    color: _scannedValue == null ? Colors.white : AppColors.statusGreen,
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
                        color: AppColors.statusAmberBg,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: AppColors.statusAmber.withValues(alpha: 0.3)),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            width: 32,
                            height: 32,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: AppColors.statusAmber.withValues(alpha: 0.2),
                            ),
                            child: const Center(
                              child: Icon(
                                Icons.lightbulb_outline,
                                color: AppColors.statusAmber,
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
                                    color: AppColors.statusAmber,
                                    fontSize: 13,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                SizedBox(height: 4),
                                Text(
                                  "Ensure the vehicle numbers are clearly visible and well-lit",
                                  style: TextStyle(
                                    color: AppColors.textGray,
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
                                color: Colors.white.withValues(alpha: 0.3),
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
                    NavButtonRow(
                      onBack: () => Navigator.pop(context),
                      onNext: _isAnalyzing ? null : _handleAnalyze,
                      nextLabel: "Analyze",
                      isNextLoading: _isAnalyzing,
                    ),
                  ],
                ),
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
              color: AppColors.darkNavySurface,
            ),
            child: Icon(
              icon,
              color: isActive ? AppColors.statusAmber : AppColors.textWhite,
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
