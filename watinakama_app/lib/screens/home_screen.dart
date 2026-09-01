import 'package:flutter/material.dart';
import 'dart:io';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_button.dart';
import '../widgets/app_card.dart';
import '../services/auth_service.dart';
import '../widgets/custom_toast.dart';
import '../services/api_service.dart';
import '../services/inspection_history_service.dart';
import '../models/inspection_record.dart';
import '../widgets/result_sheet.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AuthService _auth = AuthService();
  String _userName = "User";
  String? _profilePicPath;
  DateTime? _lastPressedAt;
  final ApiService _apiService = ApiService();
  final InspectionHistoryService _historyService = InspectionHistoryService();
  Map<String, bool> _apiStatus = {
    'engine': false,
    'body': false,
    'vin': false,
    'valuation': false,
  };
  List<InspectionRecord> _recentInspections = [];

  @override
  void initState() {
    super.initState();
    _loadUserName();
    _checkApiStatus();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final history = await _historyService.getHistory();
    if (mounted) {
      setState(() => _recentInspections = history);
    }
  }

  Future<void> _checkApiStatus() async {
    final engine = await _apiService.checkHealth(ApiConfig.engineApi);
    final body = await _apiService.checkHealth(ApiConfig.bodyApi);
    final vin = await _apiService.checkHealth(ApiConfig.vinApi);
    final valuation = await _apiService.checkHealth(ApiConfig.valuationApi);
    if (mounted) {
      setState(() {
        _apiStatus = {
          'engine': engine,
          'body': body,
          'vin': vin,
          'valuation': valuation,
        };
      });
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

  Future<void> _handleLogout() async {
    final bool? confirm = await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Logout", style: TextStyle(color: AppColors.textWhite)),
        content: const Text("Are you sure you want to logout?", style: TextStyle(color: AppColors.textGray)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel", style: TextStyle(color: AppColors.textGray)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text("Logout", style: TextStyle(color: AppColors.statusRed)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await _auth.logout();
      if (mounted) {
        Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final allServicesOnline = _apiStatus.values.every((online) => online);

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;

        final now = DateTime.now();
        final backButtonHasNotBeenPressedOrSnackBarHasBeenClosed =
            _lastPressedAt == null || now.difference(_lastPressedAt!) > const Duration(seconds: 2);

        if (backButtonHasNotBeenPressedOrSnackBarHasBeenClosed) {
          _lastPressedAt = now;
          ToastService.show(context, "Press back again to logout");
        } else {
          _handleLogout();
        }
      },
      child: Scaffold(
        backgroundColor: AppColors.darkNavyBg,
        body: SingleChildScrollView(
          child: Column(
            children: [
              WaveHeader(
                height: 140,
                child: Padding(
                  padding: const EdgeInsets.only(top: 35, right: 16, left: 16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.logout, color: AppColors.textDark, size: 20),
                        onPressed: _handleLogout,
                      ),
                      GestureDetector(
                        onTap: () async {
                          await Navigator.pushNamed(context, '/profile');
                          _loadUserName();
                        },
                        child: Row(
                          children: [
                            CircleAvatar(
                              radius: 18,
                              backgroundColor: Colors.white24,
                              backgroundImage: (_profilePicPath != null && File(_profilePicPath!).existsSync())
                                  ? FileImage(File(_profilePicPath!))
                                  : null,
                              child: (_profilePicPath == null || !File(_profilePicPath!).existsSync())
                                  ? const Icon(Icons.person, color: AppColors.textDark, size: 20)
                                  : null,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              "Hi, $_userName",
                              style: const TextStyle(
                                color: AppColors.textDark,
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          allServicesOnline ? Icons.check_circle : Icons.info_outline,
                          size: 14,
                          color: allServicesOnline ? AppColors.statusGreen : AppColors.statusAmber,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          allServicesOnline ? "All systems ready" : "Some services are still starting",
                          style: TextStyle(
                            color: allServicesOnline ? AppColors.statusGreen : AppColors.statusAmber,
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _statusDot('Engine', _apiStatus['engine']!),
                        _statusDot('Body', _apiStatus['body']!),
                        _statusDot('VIN', _apiStatus['vin']!),
                        _statusDot('Valuation', _apiStatus['valuation']!),
                      ],
                    ),
                  ],
                ),
              ),

              const Padding(
                padding: EdgeInsets.only(top: 28, bottom: 16),
                child: Text(
                  "Let's inspect your vehicle",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textWhite,
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: CustomButton(
                  text: "Start Inspection",
                  onPressed: () => Navigator.pushNamed(context, '/inspection/info'),
                ),
              ),

              const SizedBox(height: 40),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          "Recent Inspections",
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: AppColors.textWhite,
                          ),
                        ),
                        if (_recentInspections.isNotEmpty)
                          TextButton(
                            onPressed: _handleClearHistory,
                            style: TextButton.styleFrom(padding: EdgeInsets.zero),
                            child: const Text(
                              "Clear",
                              style: TextStyle(color: AppColors.textGray, fontSize: 12),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    if (_recentInspections.isEmpty)
                      AppCard(
                        padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 16),
                        child: Column(
                          children: [
                            const Icon(Icons.history, color: AppColors.textGray, size: 32),
                            const SizedBox(height: 12),
                            const Text(
                              "No inspections yet",
                              style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.w600, fontSize: 14),
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              "Completed inspections will show up here",
                              style: TextStyle(color: AppColors.textGray, fontSize: 12),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      )
                    else
                      ..._recentInspections.map(_buildInspectionCard),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _handleClearHistory() async {
    final bool? confirm = await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Clear History", style: TextStyle(color: AppColors.textWhite)),
        content: const Text(
          "This will remove all saved inspection results from this device.",
          style: TextStyle(color: AppColors.textGray),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel", style: TextStyle(color: AppColors.textGray)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text("Clear", style: TextStyle(color: AppColors.statusRed)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await _historyService.clearHistory();
      if (mounted) setState(() => _recentInspections = []);
    }
  }

  Color _verdictColor(String verdict) {
    if (verdict == 'OVERPRICED' || verdict == 'DO_NOT_BUY') return AppColors.statusRed;
    if (verdict == 'GOOD_DEAL') return AppColors.statusGreen;
    return AppColors.primaryBlue;
  }

  String _formatDate(DateTime date) {
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    return '${date.day} ${months[date.month - 1]} ${date.year}';
  }

  Widget _buildInspectionCard(InspectionRecord record) {
    final title = [record.make, record.model].where((s) => s.isNotEmpty).join(' ');
    final verdictColor = _verdictColor(record.verdict);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: GestureDetector(
        onTap: () => _showInspectionDetail(record),
        child: AppCard(
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title.isEmpty ? "Vehicle Inspection" : title,
                      style: const TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.w600, fontSize: 14),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _formatDate(record.date),
                      style: const TextStyle(color: AppColors.textGray, fontSize: 12),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    "LKR ${record.fairValueMillion.toStringAsFixed(2)}M",
                    style: const TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 14),
                  ),
                  const SizedBox(height: 4),
                  if (record.verdict.isNotEmpty)
                    Text(
                      record.verdict.replaceAll('_', ' '),
                      style: TextStyle(color: verdictColor, fontSize: 11, fontWeight: FontWeight.w600),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showInspectionDetail(InspectionRecord record) {
    final title = [record.make, record.model].where((s) => s.isNotEmpty).join(' ');
    showResultSheet(
      context,
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title.isEmpty ? "Vehicle Inspection" : title,
            style: const TextStyle(color: AppColors.textWhite, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(
            _formatDate(record.date),
            style: const TextStyle(color: AppColors.textGray, fontSize: 12),
          ),
          const SizedBox(height: 16),
          Center(
            child: Text(
              "LKR ${record.fairValueMillion.toStringAsFixed(2)}M",
              style: const TextStyle(color: AppColors.textWhite, fontSize: 28, fontWeight: FontWeight.bold),
            ),
          ),
          Center(
            child: Text(
              "Range LKR ${record.rangeMinMillion.toStringAsFixed(2)}M – LKR ${record.rangeMaxMillion.toStringAsFixed(2)}M",
              style: const TextStyle(color: AppColors.textGray, fontSize: 12),
            ),
          ),
          const SizedBox(height: 20),
          _detailRow("VIN Condition", record.vinCondition),
          _detailRow("Body Condition", "${record.bodyConditionScore.toInt()}%"),
          _detailRow("Engine Condition", "${record.engineConditionScore.toInt()}%"),
          if (record.verdict.isNotEmpty)
            _detailRow("Verdict", record.verdict.replaceAll('_', ' ')),
        ],
      ),
      ctaLabel: 'Close',
      onCta: () => Navigator.pop(context),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppColors.textGray, fontSize: 13)),
          Text(value, style: const TextStyle(color: AppColors.textWhite, fontSize: 13, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _statusDot(String label, bool isOnline) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isOnline ? AppColors.statusGreen : AppColors.statusRed,
          ),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(color: AppColors.textGray, fontSize: 10)),
      ],
    );
  }
}
