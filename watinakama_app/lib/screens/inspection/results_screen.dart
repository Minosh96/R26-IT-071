import 'package:flutter/material.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../widgets/app_card.dart';
import '../../widgets/nav_button_row.dart';
import '../../widgets/custom_button.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';
import '../../services/inspection_history_service.dart';
import '../../models/inspection_record.dart';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  Map<String, dynamic> _vehicleData = {};
  Map<String, dynamic> _valuationResult = {};
  bool _isLoading = true;
  bool _hasError = false;
  String _errorMessage = '';
  String _userName = '';
  String? _profilePicPath;
  final AuthService _auth = AuthService();
  final ApiService _apiService = ApiService();
  final InspectionHistoryService _historyService = InspectionHistoryService();
  bool _savedToHistory = false;

  // Results from all components
  double _fairValueMillion = 0;
  double _rangeMinMillion = 0;
  double _rangeMaxMillion = 0;
  double _bodyConditionScore = 0;
  double _engineConditionScore = 0;
  String _vinCondition = 'Unknown';
  String _verdict = '';
  String _explanation = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadDataAndFetch();
    });
  }

  double _parseConfidence(dynamic rawConfidence) {
    if (rawConfidence == null) return 0.0;
    if (rawConfidence is num) {
      double val = rawConfidence.toDouble();
      return val <= 1.0 ? val : val / 100.0;
    }
    if (rawConfidence is String) {
      final cleaned = rawConfidence.replaceAll('%', '').trim();
      double? parsed = double.tryParse(cleaned);
      if (parsed != null) {
        return parsed <= 1.0 ? parsed : parsed / 100.0;
      }
    }
    return 0.0;
  }

  double _parseScore(dynamic rawScore, double defaultValue) {
    if (rawScore == null) return defaultValue;
    if (rawScore is num) return rawScore.toDouble();
    if (rawScore is String) {
      return double.tryParse(rawScore.replaceAll('%', '').trim()) ?? defaultValue;
    }
    return defaultValue;
  }

  Future<void> _loadDataAndFetch() async {
    final name = await _auth.getUserName();
    final pic = await _auth.getProfilePicPath();
    
    final args = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
    if (args != null) {
      setState(() {
        _vehicleData = args; // The whole args map is the vehicle data now
        _userName = name;
        _profilePicPath = pic;
        _bodyConditionScore = _parseScore(args['body_score'], 0.0);
        _engineConditionScore = _parseScore(args['mhs_score'], _parseConfidence(args['confidence']) * 100);
        _vinCondition = (args['vin_status']?.toString().toLowerCase() == 'original') ? 'Original' : (args['vin_status'] ?? 'Unknown');
      });

    }
    _fetchValuation();
  }

  Future<void> _fetchValuation() async {
    final args = ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>;

    final requestBody = {
      'maf_year': args['maf_year'] ?? 2015,
      'reg_year': args['reg_year'] ?? 2015,
      'mileage_km': args['mileage_km'] ?? 80000,
      'previous_owners': args['previous_owners'] ?? 2,
      'is_reconditioned': args['is_reconditioned'] ?? 0,
      'power_shutters': args['power_shutters'] ?? 0,
      'power_mirrors': args['power_mirrors'] ?? 0,
      'listed_price_million': args['listed_price_million'] ?? 3.5,
      'fault_class': args['fault_class'] ?? 'healthy',
      'confidence': args['confidence'] ?? 1.0,
      'body_score': args['body_score'] ?? 100,
      'vin_status': args['vin_status'] ?? 'original',
    };

    Map<String, dynamic> result;
    try {
      result = await _apiService.getValuation(requestBody);
    } catch (_) {
      result = {'status': 'error'};
    }

    if (!mounted) return;

    if (result['status'] == 'success' || result['verdict'] != null) {
      setState(() {
        _hasError = false;
        _fairValueMillion = (result['fair_value_lkr'] ?? 0) / 1000000;
        _rangeMinMillion = (result['negotiation_min_lkr'] ?? 0) / 1000000;
        _rangeMaxMillion = (result['negotiation_max_lkr'] ?? 0) / 1000000;
        _bodyConditionScore = _parseScore(args['body_score'], 100.0);
        _engineConditionScore = _parseScore(args['mhs_score'], _parseConfidence(args['confidence']) * 100);

        // Map VIN status to display label
        String vin = args['vin_status']?.toString().toLowerCase() ?? 'unknown';
        if (vin == 'original') {
          _vinCondition = 'Original';
        } else if (vin == 'need review' || vin == 'needs review') {
          _vinCondition = 'Needs Review';
        } else if (vin == 'altered') {
          _vinCondition = 'Altered';
        } else {
          _vinCondition = vin.toUpperCase();
        }

        _verdict = result['verdict'] ?? 'FAIR_PRICE';
        _explanation = result['verdict_message'] ?? result['explanation'] ?? '';
        _isLoading = false;
        _valuationResult = result;
      });
      _saveToHistory(args);
    } else {
      // Market valuation failed - show the real VIN/body/engine results we do have,
      // but surface an explicit error instead of a made-up price.
      setState(() {
        _hasError = true;
        _errorMessage = result['message'] ?? 'Could not reach the valuation service.';
        _bodyConditionScore = _parseScore(args['body_score'], 0.0);
        _engineConditionScore = _parseScore(args['mhs_score'], _parseConfidence(args['confidence']) * 100);

        String vin = args['vin_status']?.toString().toLowerCase() ?? 'unknown';
        _vinCondition = (vin == 'original') ? 'Original' : 'Needs Review';

        _isLoading = false;
      });
    }
  }

  Future<void> _saveToHistory(Map<String, dynamic> args) async {
    if (_savedToHistory) return;
    _savedToHistory = true;

    await _historyService.addInspection(InspectionRecord(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      date: DateTime.now(),
      make: args['make']?.toString() ?? '',
      model: args['model']?.toString() ?? '',
      fairValueMillion: _fairValueMillion,
      rangeMinMillion: _rangeMinMillion,
      rangeMaxMillion: _rangeMaxMillion,
      verdict: _verdict,
      bodyConditionScore: _bodyConditionScore,
      engineConditionScore: _engineConditionScore,
      vinCondition: _vinCondition,
    ));
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
            child: const ProgressStepper(currentStep: 4),
          ),
          if (_isLoading)
            Expanded(
              child: Center(
                child: SizedBox(
                  width: 200,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: const LinearProgressIndicator(
                      color: AppColors.primaryBlue,
                      backgroundColor: AppColors.darkNavySurface,
                      minHeight: 6,
                    ),
                  ),
                ),
              ),
            )
          else
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      "Market Value Analyze",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: AppColors.textWhite,
                      ),
                    ),
                    const SizedBox(height: 16),
                    if (_hasError) _buildValuationErrorCard() else _buildFairValueCard(),
                    const SizedBox(height: 16),
                    if (!_hasError) ...[
                      _buildDepreciationFactorsCard(),
                      const SizedBox(height: 16),
                    ],
                    _buildDetailedReportCard(),
                    const SizedBox(height: 16),
                    ..._buildConditionCards(),
                    const SizedBox(height: 24),
                    NavButtonRow(
                      onBack: () => Navigator.pop(context),
                      onNext: () => Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false),
                      nextLabel: "Finish",
                    ),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildDetailedReportCard() {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Detailed Analysis Report",
            style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 12),
          _buildDetailItem(
            "Engine Fault",
            _valuationResult['engine_description'] ?? "No faults detected",
            AppColors.statusColorFor(_engineConditionScore),
          ),
          _buildDetailItem(
            "Body Damage",
            _valuationResult['body_damage_category'] != null 
              ? "Category: ${_valuationResult['body_damage_category'].toString().toUpperCase()}"
              : "No significant damage",
            AppColors.statusColorFor(_bodyConditionScore),
          ),
          _buildDetailItem(
            "VIN Status",
            _vinCondition == 'Original' ? "Legally verified" : "Verification required",
            _vinCondition == 'Original' ? AppColors.statusGreen : AppColors.statusRed,
          ),
          if (_valuationResult['vin_warning'] != null)
            Padding(
              padding: const EdgeInsets.only(top: 4, bottom: 8),
              child: Text(
                _valuationResult['vin_warning'],
                style: const TextStyle(color: AppColors.statusAmber, fontSize: 11, fontStyle: FontStyle.italic),
              ),
            ),
          const Divider(color: Colors.black12, height: 24),

          Text(
            _explanation.isNotEmpty
                ? _explanation
                : (_hasError
                    ? "VIN, body and engine analysis are complete; market pricing could not be calculated."
                    : "Analysis complete. The vehicle condition has been factored into the fair market value."),
            style: const TextStyle(color: AppColors.textGray, fontSize: 13, height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailItem(String title, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              title,
              style: const TextStyle(color: AppColors.textGray, fontSize: 12),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w500),
            ),
          ),
        ],
      ),
    );
  }


  Widget _buildValuationErrorCard() {
    return AppCard(
      padding: const EdgeInsets.all(20),
      borderColor: AppColors.statusAmber,
      child: Column(
        children: [
          const Icon(Icons.error_outline, color: AppColors.statusAmber, size: 40),
          const SizedBox(height: 12),
          const Text(
            "Market Value Unavailable",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textWhite),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            _errorMessage,
            style: const TextStyle(color: AppColors.textGray, fontSize: 13),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          CustomButton(
            text: "Retry",
            fullWidth: false,
            onPressed: () {
              setState(() => _isLoading = true);
              _fetchValuation();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildFairValueCard() {
    if (_verdict == 'DO_NOT_BUY') {
      return AppCard(
        padding: const EdgeInsets.all(20),
        borderColor: AppColors.statusRed,
        child: Column(
          children: [
            const Icon(
              Icons.warning_amber_rounded,
              color: AppColors.statusRed,
              size: 48,
            ),
            const SizedBox(height: 12),
            const Text(
              "VALUATION BLOCKED",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: AppColors.statusRed,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              _explanation.isNotEmpty ? _explanation : "This vehicle is blocked from valuation due to critical compliance/security risks.",
              style: const TextStyle(color: AppColors.textGray, fontSize: 13),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            _buildVerdictChip(),
          ],
        ),
      );
    }

    return AppCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          const Text(
            "Estimated Fair Value",
            style: TextStyle(color: AppColors.textGray, fontSize: 13),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            "LKR ${_fairValueMillion.toStringAsFixed(2)}M",
            style: const TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: AppColors.textWhite,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 6),
          Text(
            "Range LKR ${_rangeMinMillion}M  –  LKR ${_rangeMaxMillion}M",
            style: const TextStyle(color: AppColors.textGray, fontSize: 13),
            textAlign: TextAlign.center,
          ),
          if (_verdict.isNotEmpty) ...[
            const SizedBox(height: 12),
            _buildVerdictChip(),
          ],
        ],
      ),
    );
  }

  Widget _buildVerdictChip() {
    Color bgColor;
    Color textColor;
    String label = _verdict.replaceAll('_', ' ');

    if (_verdict == 'OVERPRICED' || _verdict == 'DO_NOT_BUY') {
      bgColor = AppColors.statusRed.withValues(alpha: 0.2);
      textColor = AppColors.statusRed;
    } else if (_verdict == 'GOOD_DEAL') {
      bgColor = AppColors.statusGreen.withValues(alpha: 0.2);
      textColor = AppColors.statusGreen;
    } else {
      bgColor = AppColors.primaryBlue.withValues(alpha: 0.2);
      textColor = AppColors.primaryBlue;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: textColor,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      ),
    );
  }

  Widget _buildDepreciationFactorsCard() {
    final args = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};
    
    // Calculate percentages based on deductions or scores
    int bodyDep = (_valuationResult['body_deduction_lkr'] != null && _fairValueMillion > 0)
        ? -((_valuationResult['body_deduction_lkr'] as num) / (_fairValueMillion * 1000000) * 100).round()
        : -((100 - _bodyConditionScore) / 5).round();
        
    int mechDep = (_valuationResult['engine_deduction_lkr'] != null && _fairValueMillion > 0)
        ? -((_valuationResult['engine_deduction_lkr'] as num) / (_fairValueMillion * 1000000) * 100).round()
        : -((100 - _engineConditionScore) / 5).round();

    // Estimate mileage and year depreciation
    int mileageKm = args['mileage_km'] ?? 80000;
    int mileageDep = -(mileageKm / 10000).round();
    
    int year = args['maf_year'] ?? 2015;
    int yearDep = -(2026 - year);

    final factors = [
      {'label': 'Body Condition', 'value': bodyDep, 'color': AppColors.statusColorFor(_bodyConditionScore)},
      {'label': 'Mileage', 'value': mileageDep, 'color': AppColors.statusAmber},
      {'label': 'Mechanical', 'value': mechDep, 'color': AppColors.statusColorFor(_engineConditionScore)},
      {'label': 'Year', 'value': yearDep, 'color': AppColors.primaryBlue},
    ];


    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Depreciation Factors",
            style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 14),
          ...factors.map((f) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  children: [
                    SizedBox(
                      width: 130,
                      child: Text(
                        f['label'] as String,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(color: AppColors.textWhite, fontSize: 12),
                      ),
                    ),
                    Expanded(
                      child: Stack(
                        children: [
                          Container(
                            height: 8,
                            decoration: BoxDecoration(
                              color: Colors.grey.shade300,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                          FractionallySizedBox(
                            widthFactor: ((f['value'] as int).abs() / 20.0).clamp(0.0, 1.0),
                            child: Container(
                              height: 8,
                              decoration: BoxDecoration(
                                color: f['color'] as Color,
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      "${f['value']}%",
                      style: const TextStyle(color: AppColors.textGray, fontSize: 12),
                    ),
                  ],
                ),
              )),
        ],
      ),
    );
  }

  List<Widget> _buildConditionCards() {
    final data = [
      {'label': 'VIN Condition', 'display': _vinCondition, 'type': 'vin'},
      {
        'label': 'Body Condition',
        'display': '${_bodyConditionScore.toInt()}%',
        'type': 'score',
        'score': _bodyConditionScore
      },
      {
        'label': 'Engine Condition',
        'display': '${_engineConditionScore.toInt()}%',
        'type': 'score',
        'score': _engineConditionScore
      },
    ];

    return data.map((item) {
      Color valueColor;
      if (item['type'] == 'vin') {
        valueColor = item['display'] == 'Original'
            ? AppColors.statusGreen
            : (item['display'] == 'Needs Review' ? AppColors.statusAmber : AppColors.statusRed);
      } else {
        valueColor = AppColors.statusColorFor((item['score'] as num).toDouble());
      }

      return Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: AppCard(
          radius: 12,
          padding: const EdgeInsets.all(16),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                item['label'] as String,
                style: const TextStyle(color: AppColors.textWhite, fontSize: 14),
              ),
              Text(
                item['display'] as String,
                style: TextStyle(color: valueColor, fontWeight: FontWeight.bold, fontSize: 14),
              ),
            ],
          ),
        ),
      );
    }).toList();
  }
}

