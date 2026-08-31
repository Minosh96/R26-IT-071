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

// ─── Vehicle view names in API / model order ─────────────────────────────────
const List<String> _kAngleNames  = ['Front', 'Rear', 'Left', 'Right', 'Up'];
const List<String> _kApiViewKeys = ['front', 'rear', 'left', 'right', 'roof'];

class BodyImagesScreen extends StatefulWidget {
  const BodyImagesScreen({super.key});

  @override
  State<BodyImagesScreen> createState() => _BodyImagesScreenState();
}

class _BodyImagesScreenState extends State<BodyImagesScreen> {
  Map<String, dynamic> _vehicleData = {};
  String _userName = '';
  String? _profilePicPath;

  /// Index of the view the user is currently looking at / about to capture.
  int _currentAngle = 0;

  /// Captured image paths (null = not yet captured).
  final List<String?> _capturedPaths = [null, null, null, null, null];

  /// Per-image view-validation state.
  /// null  = not validated yet
  /// true  = validated OK
  /// false = wrong view detected
  final List<bool?>         _viewValid      = [null, null, null, null, null];
  final List<String?>       _viewMessage    = [null, null, null, null, null];
  final List<bool>          _viewValidating = [false, false, false, false, false];

  final AuthService _authService = AuthService();
  final ApiService  _apiService  = ApiService();

  bool _isAnalyzing = false;

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  Future<void> _loadInitialData() async {
    final name = await _authService.getUserName();
    final pic  = await _authService.getProfilePicPath();
    setState(() {
      _userName    = name;
      _profilePicPath = pic;
    });

    WidgetsBinding.instance.addPostFrameCallback((_) {
      final args = ModalRoute.of(context)?.settings.arguments;
      if (args is Map<String, dynamic>) {
        setState(() => _vehicleData = args);
      }
    });
  }

  // ── Computed helpers ────────────────────────────────────────────────────────

  List<bool> get _capturedAngles => _capturedPaths.map((p) => p != null).toList();
  bool get _allCaptured => _capturedPaths.every((p) => p != null);

  String get _currentAngleName => _kAngleNames[_currentAngle];

  /// Returns true if current image has a validation error (wrong view).
  bool get _currentHasError =>
      _viewValid[_currentAngle] == false &&
      _viewMessage[_currentAngle] != null;

  // ── Image capture & validation ──────────────────────────────────────────────

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final photo  = await picker.pickImage(source: source, imageQuality: 85);
    if (photo == null) return;

    setState(() {
      _capturedPaths[_currentAngle]  = photo.path;
      _viewValid[_currentAngle]      = null;   // reset validation
      _viewMessage[_currentAngle]    = null;
      _viewValidating[_currentAngle] = true;
    });

    // Validate immediately in background
    await _validateCurrentView(_currentAngle, photo.path);

    // Auto-advance to next un-captured angle (only if view is OK)
    if (mounted && _viewValid[_currentAngle] != false && !_allCaptured) {
      for (int i = 0; i < 5; i++) {
        if (_capturedPaths[i] == null) {
          setState(() => _currentAngle = i);
          break;
        }
      }
    }
  }

  Future<void> _validateCurrentView(int index, String imagePath) async {
    final expectedView = _kApiViewKeys[index];
    final result = await _apiService.validateView(imagePath, expectedView);

    if (!mounted) return;
    setState(() {
      _viewValidating[index] = false;
      final isCorrect = result['correct'] as bool? ?? true;
      final isUncertain = result['uncertain'] as bool? ?? false;
      _viewValid[index]   = isCorrect || isUncertain;   // uncertain = soft-pass
      _viewMessage[index] = result['message'] as String?;
    });

    // Show a toast for wrong views
    if (mounted && _viewValid[index] == false) {
      final predicted = result['predicted'] as String? ?? 'unknown';
      ToastService.show(
        context,
        '⚠️ This looks like a $predicted view.\n'
        'Please upload the ${_kAngleNames[index]} view image.',
        isError: true,
      );
    }
  }

  // ── Submit / Analyze ────────────────────────────────────────────────────────

  Future<void> _handleNext() async {
    // Check for any images with confirmed wrong-view errors
    final badViews = <String>[];
    for (int i = 0; i < 5; i++) {
      if (_capturedPaths[i] != null && _viewValid[i] == false) {
        badViews.add(_kAngleNames[i]);
      }
    }

    if (badViews.isNotEmpty) {
      ToastService.show(
        context,
        'Please fix these views before continuing:\n${badViews.join(', ')}',
        isError: true,
      );
      return;
    }

    final images = _capturedPaths
        .where((p) => p != null)
        .map((p) => File(p!))
        .toList();

    if (images.isEmpty) {
      Navigator.pushNamed(context, '/inspection/vin', arguments: {
        ..._vehicleData,
        'body_score': null,
        'body_images': _capturedPaths,
        'body_damages': const [],
      });
      return;
    }

    setState(() => _isAnalyzing = true);

    final result = await _apiService.analyzeBody(images);

    if (!mounted) return;
    setState(() => _isAnalyzing = false);

    if (result['status'] == 'error' || result['status_code'] == 503) {
      String msg = result['message'] ?? 'Body analysis service is unavailable.';
      if (result['status_code'] == 503) {
        msg = 'Body models are still loading or missing. Please wait a moment.';
      }
      ToastService.show(context, msg, isError: true);
      return;
    }

    // Check if the backend reported any view errors
    final viewErrors = (result['view_errors'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    if (viewErrors.isNotEmpty) {
      _showViewErrorSheet(viewErrors, result);
    } else {
      _showBodyResult(result);
    }
  }

  // ── Result / Error sheets ───────────────────────────────────────────────────

  /// Shown when backend detected at least one wrong view in the uploaded images.
  void _showViewErrorSheet(
    List<Map<String, dynamic>> viewErrors,
    Map<String, dynamic> fullResult,
  ) {
    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Color(0xFFFF6B35), size: 28),
              SizedBox(width: 10),
              Text(
                'Wrong View Detected',
                style: TextStyle(
                  color: AppColors.textWhite,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            'The AI model detected that some images do not match '
            'the expected vehicle view. Please re-upload the correct images:',
            style: TextStyle(color: AppColors.textGray, fontSize: 13),
          ),
          const SizedBox(height: 16),
          ...viewErrors.map((e) => _buildViewErrorTile(e)),
          const SizedBox(height: 16),
          if ((fullResult['damaged_parts'] as List?)?.isNotEmpty == true) ...[
            const Text(
              'Partial results (valid views only):',
              style: TextStyle(color: AppColors.textGray, fontSize: 12),
            ),
            const SizedBox(height: 8),
            _buildScoreChip(
              (fullResult['final_body_condition_score'] ??
                  fullResult['body_score'] ??
                  fullResult['score'] ??
                  0).toDouble(),
            ),
          ],
        ],
      ),
      ctaLabel: 'Retake Wrong Image(s)',
      onCta: () => Navigator.pop(context),
    );
  }

  Widget _buildViewErrorTile(Map<String, dynamic> e) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0x1FFF4444),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0x66FF4444)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.close_rounded, color: AppColors.statusRed, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${e['view']?.toString().toUpperCase() ?? 'Unknown'} View',
                  style: const TextStyle(
                    color: AppColors.textWhite,
                    fontWeight: FontWeight.bold,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Expected: ${e['expected'] ?? '—'}  |  Detected: ${e['detected'] ?? '—'}',
                  style: const TextStyle(color: AppColors.textGray, fontSize: 12),
                ),
                const SizedBox(height: 4),
                Text(
                  e['message'] ?? '',
                  style: const TextStyle(color: AppColors.textGray, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Full result sheet – body condition score + damaged body parts.
  void _showBodyResult(Map<String, dynamic> result) {
    final double score = (result['final_body_condition_score'] ??
            result['body_score'] ??
            result['score'] ??
            0)
        .toDouble();

    // Collect damaged_parts list from backend
    final rawDamages =
        (result['damaged_parts'] as List?)?.cast<Map<String, dynamic>>() ?? [];

    final color = AppColors.statusColorFor(score);

    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Score circle + label ──────────────────────────────────────────
          Center(
            child: Column(
              children: [
                Text(
                  '${score.toInt()}',
                  style: TextStyle(
                    fontSize: 56,
                    fontWeight: FontWeight.bold,
                    color: color,
                    height: 1,
                  ),
                ),
                const Text(
                  'Body Condition Score',
                  style: TextStyle(color: AppColors.textGray, fontSize: 13),
                ),
                const SizedBox(height: 8),
                Text(
                  result['condition'] ?? result['vehicle_status'] ?? '',
                  style: TextStyle(
                    color: color,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // ── Score bar ─────────────────────────────────────────────────────
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: score / 100,
              backgroundColor: Colors.black26,
              color: color,
              minHeight: 10,
            ),
          ),
          const SizedBox(height: 24),

          // ── Damaged body parts ────────────────────────────────────────────
          if (rawDamages.isNotEmpty) ...[
            const Text(
              'Damaged Body Parts',
              style: TextStyle(
                color: AppColors.textWhite,
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 10),
            ...rawDamages.map((d) => _buildDamageTile(d)),
          ] else ...[
            const Center(
              child: Text(
                '✅  No significant body damage detected.',
                style: TextStyle(color: AppColors.statusGreen, fontSize: 13),
              ),
            ),
          ],
        ],
      ),
      ctaLabel: 'Continue to VIN Scan',
      onCta: () {
        Navigator.pop(context);
        Navigator.pushNamed(
          context,
          '/inspection/vin',
          arguments: {
            ..._vehicleData,
            'body_score': score,
            'body_images': _capturedPaths,
            'body_damages': rawDamages
                .map((d) => '${d['part']} – ${d['damage_type']}')
                .toList(),
          },
        );
      },
    );
  }

  Widget _buildDamageTile(Map<String, dynamic> d) {
    final part       = (d['part'] ?? '—').toString().replaceAll('_', ' ');
    final damageType = d['damage_type']?.toString() ?? '—';
    final view       = d['view']?.toString() ?? '';
    final category   = d['category']?.toString() ?? '';

    Color dtColor;
    IconData dtIcon;
    switch (damageType.toLowerCase()) {
      case 'dent':
        dtColor = const Color(0xFFFF6B35);
        dtIcon  = Icons.format_align_justify;
        break;
      case 'rust':
        dtColor = const Color(0xFFD4622A);
        dtIcon  = Icons.grain;
        break;
      case 'scratch':
        dtColor = const Color(0xFFFFCC00);
        dtIcon  = Icons.swipe;
        break;
      default:
        dtColor = AppColors.textGray;
        dtIcon  = Icons.info_outline;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.darkNavySurface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.textFieldBorder),
      ),
      child: Row(
        children: [
          Icon(dtIcon, color: dtColor, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  part,
                  style: const TextStyle(
                    color: AppColors.textWhite,
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '$damageType${category.isNotEmpty ? '  ·  $category' : ''}${view.isNotEmpty ? '  ·  ${view[0].toUpperCase()}${view.substring(1)} view' : ''}',
                  style: TextStyle(color: dtColor, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScoreChip(double score) {
    final color = AppColors.statusColorFor(score);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(
        'Partial Score: ${score.toInt()}/100',
        style: TextStyle(color: color, fontWeight: FontWeight.bold),
      ),
    );
  }

  // ── Build ───────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final hasError  = _currentHasError;
    final isValidating = _viewValidating[_currentAngle];

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
            // ── Step progress ──────────────────────────────────────────────
            Container(
              height: 75,
              width: double.infinity,
              color: AppColors.lightBlueTop,
              alignment: Alignment.center,
              child: const ProgressStepper(currentStep: 2),
            ),

            const SizedBox(height: 12),

            // ── Sub-stepper (angle selector) ───────────────────────────────
            ImageSubStepper(
              currentAngle: _currentAngle,
              capturedAngles: _capturedAngles,
              onAngleSelected: (index) => setState(() => _currentAngle = index),
            ),

            const SizedBox(height: 16),

            // ── Header + validation status ─────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                children: [
                  Text(
                    _capturedPaths[_currentAngle] == null
                        ? 'Capture the $_currentAngleName view'
                        : '$_currentAngleName view captured',
                    style: const TextStyle(
                      color: AppColors.textWhite,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 6),

                  // Validation status banner
                  if (isValidating)
                    _buildStatusBanner(
                      icon: Icons.hourglass_top_rounded,
                      color: AppColors.textGray,
                      text: 'Verifying view...',
                    )
                  else if (hasError)
                    _buildStatusBanner(
                      icon: Icons.error_outline_rounded,
                      color: AppColors.statusRed,
                      text: _viewMessage[_currentAngle] ?? 'Wrong view detected.',
                      isError: true,
                    )
                  else if (_viewValid[_currentAngle] == true &&
                      _capturedPaths[_currentAngle] != null)
                    _buildStatusBanner(
                      icon: Icons.check_circle_outline_rounded,
                      color: AppColors.statusGreen,
                      text: 'View verified ✓',
                    ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // ── Preview area ───────────────────────────────────────────────
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: AppColors.darkNavySurface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: hasError
                          ? AppColors.statusRed.withValues(alpha: 0.7)
                          : (_viewValid[_currentAngle] == true
                              ? AppColors.statusGreen.withValues(alpha: 0.5)
                              : AppColors.textFieldBorder),
                      width: hasError || _viewValid[_currentAngle] == true ? 2 : 1,
                    ),
                  ),
                  child: _capturedPaths[_currentAngle] != null
                      ? Stack(
                          fit: StackFit.expand,
                          children: [
                            ClipRRect(
                              borderRadius: BorderRadius.circular(11),
                              child: Image.file(
                                File(_capturedPaths[_currentAngle]!),
                                fit: BoxFit.cover,
                              ),
                            ),
                            // Retake hint
                            Positioned(
                              top: 12,
                              right: 12,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 4),
                                decoration: BoxDecoration(
                                  color: Colors.black54,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Text(
                                  'Tap Capture to Retake',
                                  style: TextStyle(
                                      color: Colors.white70, fontSize: 10),
                                ),
                              ),
                            ),
                            // View-validation overlay badge
                            if (isValidating)
                              Positioned(
                                bottom: 12,
                                left: 12,
                                child: _buildBadge(
                                    Icons.hourglass_top_rounded,
                                    'Verifying view...',
                                    AppColors.textGray),
                              )
                            else if (hasError)
                              Positioned(
                                bottom: 12,
                                left: 12,
                                right: 12,
                                child: _buildBadge(
                                    Icons.error_rounded,
                                    'Wrong view – please retake',
                                    AppColors.statusRed),
                              )
                            else if (_viewValid[_currentAngle] == true)
                              Positioned(
                                bottom: 12,
                                left: 12,
                                child: _buildBadge(
                                    Icons.check_circle_rounded,
                                    'View verified',
                                    AppColors.statusGreen),
                              ),
                          ],
                        )
                      : Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.camera_alt_outlined,
                                color: AppColors.textGray, size: 48),
                            const SizedBox(height: 12),
                            Text(
                              'Tap Capture for $_currentAngleName view',
                              style: const TextStyle(
                                  color: AppColors.textGray, fontSize: 13),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              'The AI will verify this is the correct view',
                              style: TextStyle(
                                  color: AppColors.textGray
                                      .withValues(alpha: 0.6),
                                  fontSize: 11),
                            ),
                          ],
                        ),
                ),
              ),
            ),

            const SizedBox(height: 16),

            // ── Camera / Gallery buttons ───────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildActionButton(
                    icon: Icons.camera_alt,
                    label: _capturedPaths[_currentAngle] == null
                        ? 'Capture'
                        : 'Retake',
                    onTap: () => _pickImage(ImageSource.camera),
                    isCamera: true,
                  ),
                  _buildActionButton(
                    icon: Icons.photo_library_outlined,
                    label: 'Gallery',
                    onTap: () => _pickImage(ImageSource.gallery),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // ── Nav buttons ────────────────────────────────────────────────
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

  // ── Helper widgets ──────────────────────────────────────────────────────────

  Widget _buildStatusBanner({
    required IconData icon,
    required Color color,
    required String text,
    bool isError = false,
  }) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: isError ? 0.12 : 0.08),
        borderRadius: BorderRadius.circular(8),
        border: isError ? Border.all(color: color.withValues(alpha: 0.4)) : null,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(color: color, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBadge(IconData icon, String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.75),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.6)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 6),
          Text(text, style: TextStyle(color: color, fontSize: 11)),
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
            width:  isCamera ? 64 : 52,
            height: isCamera ? 64 : 52,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isCamera ? AppColors.primaryBlue : AppColors.darkNavySurface,
              border: isCamera
                  ? Border.all(
                      color: Colors.white.withValues(alpha: 0.3),
                      width: 3,
                    )
                  : null,
            ),
            child: Icon(
              icon,
              color: isCamera ? Colors.white : AppColors.textWhite,
              size:  isCamera ? 30 : 24,
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
