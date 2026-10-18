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

const String _vinIssueMessage =
    "There is an issue with this vehicle's VIN number. Please check this vehicle further before buying.";
const String _vinIssueAdvice =
    "Compare the VIN with the registration book, or ask the RMV or a trusted mechanic to verify it.";

class _ResultsScreenState extends State<ResultsScreen> {
  Map<String, dynamic> _vehicleData = {};
  Map<String, dynamic> _valuationResult = {};
  bool _isLoading = true;
  bool _hasError = false;
  bool _outOfScope = false;
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
      // The engine deduction is based on the health score; fault_class is only a fallback
      'mhs_score': args['mhs_score'],
      'body_score': args['body_score'] ?? 100,
      'vin_status': args['vin_status'] ?? 'unknown',
      // Lets the valuation service refuse cars it was not trained on
      'make': args['make'],
      'model': args['model'],
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
        // The range where 80% of comparable real listings fall
        _rangeMinMillion = (result['fair_low_lkr'] ?? result['negotiation_min_lkr'] ?? 0) / 1000000;
        _rangeMaxMillion = (result['fair_high_lkr'] ?? result['negotiation_max_lkr'] ?? 0) / 1000000;
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
        // Altered VIN: always show the friendly wording, even if an older
        // valuation service still answers DO_NOT_BUY / "Do not purchase".
        if (_verdict == 'DO_NOT_BUY' || _verdict == 'CHECK_VIN') {
          _verdict = 'CHECK_VIN';
          _explanation = _vinIssueMessage;
        }
        _isLoading = false;
        _valuationResult = result;
      });
      _saveToHistory(args);
    } else {
      // Market valuation failed - show the real VIN/body/engine results we do have,
      // but surface an explicit error instead of a made-up price.
      setState(() {
        _hasError = true;
        _outOfScope = result['status'] == 'out_of_scope';
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
                    // No price breakdown when the VIN blocks the valuation
                    if (!_hasError && _verdict != 'CHECK_VIN') ...[
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
    // Drive every line from the ACTUAL per-component outputs (carried in
    // _vehicleData) — not from optional valuation fields that silently fall
    // back to "no problem" and misreport a damaged/altered vehicle.

    // Engine — reflect the real fault class from the engine component.
    final faultClass = (_vehicleData['fault_class'] ?? '').toString();
    final bool engineAnalyzed = faultClass.isNotEmpty || _engineConditionScore > 0;
    final bool engineHealthy = faultClass.isEmpty || faultClass.toLowerCase() == 'healthy';
    String engineText;
    Color engineColor;
    if (!engineAnalyzed) {
      engineText = "Not analyzed";
      engineColor = AppColors.textGray;
    } else if (engineHealthy) {
      engineText = "No faults detected";
      engineColor = AppColors.statusGreen;
    } else {
      engineText = "${_prettifyFault(faultClass)} detected";
      engineColor = AppColors.statusColorFor(_engineConditionScore);
    }

    // Body — reflect the real detected damages / condition score.
    final bodyDamages = (_vehicleData['body_damages'] as List?) ?? const [];
    final bool bodyAnalyzed = _vehicleData['body_score'] != null || bodyDamages.isNotEmpty;
    String bodyText;
    Color bodyColor;
    if (!bodyAnalyzed) {
      bodyText = "Not analyzed";
      bodyColor = AppColors.textGray;
    } else if (bodyDamages.isNotEmpty) {
      bodyText = "${bodyDamages.length} issue${bodyDamages.length == 1 ? '' : 's'} detected";
      bodyColor = AppColors.statusColorFor(_bodyConditionScore);
    } else {
      bodyText = "No significant damage";
      bodyColor = AppColors.statusColorFor(_bodyConditionScore);
    }

    // VIN — surface the real verdict instead of collapsing everything
    // non-original into a soft "Verification required".
    String vinText;
    Color vinColor;
    switch (_vinCondition) {
      case 'Original':
        vinText = "Legally verified";
        vinColor = AppColors.statusGreen;
        break;
      case 'Altered':
        vinText = "VIN issue — check further before buying";
        vinColor = AppColors.statusAmber;
        break;
      case 'Needs Review':
        vinText = "Verification required";
        vinColor = AppColors.statusAmber;
        break;
      default:
        vinText = "Not verified";
        vinColor = AppColors.textGray;
    }

    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Detailed Analysis Report",
            style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 12),
          _buildDetailItem("Engine Fault", engineText, engineColor),
          _buildDetailItem("Body Damage", bodyText, bodyColor),
          _buildDetailItem("VIN Status", vinText, vinColor),
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

  /// Turns a raw fault class like "battery_fault" into "Battery fault".
  String _prettifyFault(String faultClass) {
    final cleaned = faultClass.replaceAll('_', ' ').trim();
    if (cleaned.isEmpty) return 'Fault';
    return cleaned[0].toUpperCase() + cleaned.substring(1);
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
          Text(
            _outOfScope ? "Can't Value This Car Reliably" : "Market Value Unavailable",
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textWhite),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            _errorMessage,
            style: const TextStyle(color: AppColors.textGray, fontSize: 13),
            textAlign: TextAlign.center,
          ),
          if (!_outOfScope) ...[
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
        ],
      ),
    );
  }

  Widget _buildFairValueCard() {
    // Altered VIN: no price until the VIN is checked (DO_NOT_BUY is the older name)
    if (_verdict == 'CHECK_VIN' || _verdict == 'DO_NOT_BUY') {
      return AppCard(
        padding: const EdgeInsets.all(20),
        borderColor: AppColors.statusAmber,
        child: Column(
          children: [
            const Icon(
              Icons.warning_amber_rounded,
              color: AppColors.statusAmber,
              size: 48,
            ),
            const SizedBox(height: 12),
            const Text(
              "VIN Needs Checking",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: AppColors.statusAmber,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              _vinIssueMessage,
              style: TextStyle(color: AppColors.textWhite, fontSize: 14),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              _vinIssueAdvice,
              style: TextStyle(color: AppColors.textGray, fontSize: 12),
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
            "Fair range LKR ${_rangeMinMillion.toStringAsFixed(2)}M  –  LKR ${_rangeMaxMillion.toStringAsFixed(2)}M",
            style: const TextStyle(color: AppColors.textGray, fontSize: 13),
            textAlign: TextAlign.center,
          ),
          if (_valuationResult['recommended_offer_lkr'] != null && _verdict != 'GOOD_DEAL') ...[
            const SizedBox(height: 6),
            Text(
              "Recommended offer LKR ${_millions(_valuationResult['recommended_offer_lkr'])}M "
              "(negotiate ${_millions(_valuationResult['negotiation_min_lkr'])}M – ${_millions(_valuationResult['negotiation_max_lkr'])}M)",
              style: const TextStyle(color: AppColors.textWhite, fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ],
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

    if (_verdict == 'CHECK_VIN' || _verdict == 'DO_NOT_BUY') {
      bgColor = AppColors.statusAmber.withValues(alpha: 0.2);
      textColor = AppColors.statusAmber;
    } else if (_verdict == 'OVERPRICED') {
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

  String _millions(dynamic lkr) => ((lkr as num? ?? 0) / 1000000).toStringAsFixed(2);

  /// Depreciation factors as a bar chart. Each bar is that factor's effect in
  /// rupees from the valuation service, shown as a % of the market value:
  ///   Body / Mechanical - repair cost from the survey (body score, engine health)
  ///   Mileage / Year    - the model's price effect compared with a typical Alto
  Widget _buildDepreciationFactorsCard() {
    final r = _valuationResult;
    final market = (r['base_market_value_lkr'] as num? ?? 0).toDouble();
    final factors = (r['price_factors'] as List? ?? []).cast<Map>();

    double factorLkr(String key) {
      final match = factors.where((f) => f['factor'] == key);
      return match.isEmpty ? 0 : (match.first['impact_lkr'] as num).toDouble();
    }

    double pct(double lkr) => market > 0 ? lkr / market * 100 : 0;

    final rows = <Map<String, dynamic>>[
      {'label': 'Body Condition', 'value': pct(-(r['body_deduction_lkr'] as num? ?? 0).toDouble()), 'color': AppColors.statusColorFor(_bodyConditionScore)},
      {'label': 'Mileage', 'value': pct(factorLkr('mileage')), 'color': AppColors.statusAmber},
      {'label': 'Mechanical', 'value': pct(-(r['engine_deduction_lkr'] as num? ?? 0).toDouble()), 'color': AppColors.statusColorFor(_engineConditionScore)},
      {'label': 'Year', 'value': pct(factorLkr('year')), 'color': AppColors.primaryBlue},
      if ((r['vin_discount_lkr'] as num? ?? 0) > 0)
        {'label': 'VIN Not Confirmed', 'value': pct(-(r['vin_discount_lkr'] as num).toDouble()), 'color': AppColors.statusRed},
    ];

    // The largest effect fills the bar; at least 5% so small effects stay small.
    final scale = rows.map((f) => (f['value'] as double).abs()).fold<double>(5, (a, b) => b > a ? b : a);

    String label(double v) {
      final shown = v.abs() < 10 ? v.toStringAsFixed(1) : v.toStringAsFixed(0);
      return v > 0 ? '+$shown%' : '$shown%';
    }

    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Depreciation Factors",
            style: TextStyle(color: AppColors.textWhite, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 14),
          ...rows.map((f) => Padding(
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
                            widthFactor: ((f['value'] as double).abs() / scale).clamp(0.0, 1.0),
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
                    SizedBox(
                      width: 48,
                      child: Text(
                        label(f['value'] as double),
                        textAlign: TextAlign.right,
                        style: const TextStyle(color: AppColors.textGray, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              )),
          if (r['market_data_warning'] != null)
            Text(
              r['market_data_warning'],
              style: const TextStyle(color: AppColors.statusAmber, fontSize: 11, fontStyle: FontStyle.italic),
            ),
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

