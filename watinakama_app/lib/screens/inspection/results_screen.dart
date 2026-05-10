import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';
import 'dart:io';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  Map<String, dynamic> _vehicleData = {};
  // ignore: unused_field
  Map<String, dynamic> _valuationResult = {};
  bool _isLoading = true;
  String _userName = '';
  String? _profilePicPath;
  final AuthService _auth = AuthService();
  final ApiService _apiService = ApiService();

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

  Future<void> _loadDataAndFetch() async {
    final name = await _auth.getUserName();
    final pic = await _auth.getProfilePicPath();
    
    final args = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
    if (args != null) {
      setState(() {
        _vehicleData = args; // The whole args map is the vehicle data now
        _userName = name;
        _profilePicPath = pic;
        _bodyConditionScore = (args['body_score'] ?? 0).toDouble();
        _engineConditionScore = (args['mhs_score'] ?? (args['confidence'] ?? 0).toDouble() * 100).toDouble();
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

    final result = await _apiService.getValuation(requestBody);

    if (result['status'] == 'success' || result['verdict'] != null) {
      setState(() {
        _fairValueMillion = (result['fair_value_lkr'] ?? 0) / 1000000;
        _rangeMinMillion = (result['negotiation_min_lkr'] ?? 0) / 1000000;
        _rangeMaxMillion = (result['negotiation_max_lkr'] ?? 0) / 1000000;
        _bodyConditionScore = (args['body_score'] ?? 100).toDouble();
        _engineConditionScore = (args['mhs_score'] ?? args['confidence']?.toDouble() * 100 ?? 100).toDouble();
        
        // Map VIN status to display label
        String vin = args['vin_status']?.toString().toLowerCase() ?? 'unknown';
        if (vin == 'original') _vinCondition = 'Original';
        else if (vin == 'need review' || vin == 'needs review') _vinCondition = 'Needs Review';
        else if (vin == 'altered') _vinCondition = 'Altered';
        else _vinCondition = vin.toUpperCase();

        _verdict = result['verdict'] ?? 'FAIR_PRICE';
        _explanation = result['explanation'] ?? '';
        _isLoading = false;
        _valuationResult = result;
      });
    } else {
      // Use demo data for prices if API fails, but keep ACTUAL analysis results for scores
      setState(() {
        _fairValueMillion = 3.85;
        _rangeMinMillion = 3.6;
        _rangeMaxMillion = 4.1;
        _bodyConditionScore = (args['body_score'] ?? 0).toDouble();
        _engineConditionScore = (args['mhs_score'] ?? (args['confidence'] ?? 0) * 100).toDouble();
        
        String vin = args['vin_status']?.toString().toLowerCase() ?? 'unknown';
        _vinCondition = (vin == 'original') ? 'Original' : 'Needs Review';
        
        _isLoading = false;
      });
    }


  }

  void _usePlaceholders() {
    setState(() {
      _fairValueMillion = 3.85;
      _rangeMinMillion = 3.6;
      _rangeMaxMillion = 4.1;
      _bodyConditionScore = 80;
      _engineConditionScore = 50;
      _vinCondition = 'Original';
      _verdict = 'FAIR_PRICE';
      _isLoading = false;
    });
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
            const Expanded(
              child: Center(
                child: CircularProgressIndicator(color: AppColors.primaryBlue),
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
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildFairValueCard(),
                    const SizedBox(height: 16),
                    _buildDepreciationFactorsCard(),
                    const SizedBox(height: 16),
                    _buildDetailedReportCard(),
                    const SizedBox(height: 16),
                    ..._buildConditionCards(),
                    const SizedBox(height: 24),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        ElevatedButton(
                          onPressed: () => Navigator.pop(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF1A1A2E),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(24),
                            ),
                          ),
                          child: const Text("Back"),
                        ),
                        ElevatedButton(
                          onPressed: () => Navigator.pushNamedAndRemoveUntil(
                            context, 
                            '/home', 
                            (route) => false
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2E7D32),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(24),
                            ),
                          ),
                          child: const Text("Finish"),
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
    );
  }

  Widget _buildDetailedReportCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2035),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Detailed Analysis Report",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 12),
          _buildDetailItem(
            "Engine Fault",
            _valuationResult['engine_description'] ?? "No faults detected",
            _getScoreColor(_engineConditionScore),
          ),
          _buildDetailItem(
            "Body Damage",
            _valuationResult['body_damage_category'] != null 
              ? "Category: ${_valuationResult['body_damage_category'].toString().toUpperCase()}"
              : "No significant damage",
            _getScoreColor(_bodyConditionScore),
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
          const Divider(color: Colors.white10, height: 24),

          Text(
            _explanation.isNotEmpty ? _explanation : "Analysis complete. The vehicle condition has been factored into the fair market value.",
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
              style: const TextStyle(color: Colors.white70, fontSize: 12),
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


  Widget _buildFairValueCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2035),
        borderRadius: BorderRadius.circular(16),
      ),
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
              color: Colors.white,
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

    if (_verdict == 'OVERPRICED') {
      bgColor = AppColors.statusRed.withOpacity(0.2);
      textColor = AppColors.statusRed;
    } else if (_verdict == 'GOOD_DEAL') {
      bgColor = AppColors.statusGreen.withOpacity(0.2);
      textColor = AppColors.statusGreen;
    } else {
      bgColor = AppColors.primaryBlue.withOpacity(0.2);
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
    int bodyDep = _valuationResult['body_deduction_lkr'] != null 
        ? -((_valuationResult['body_deduction_lkr'] as num) / (_fairValueMillion * 1000000) * 100).round()
        : -((100 - _bodyConditionScore) / 5).round();
        
    int mechDep = _valuationResult['engine_deduction_lkr'] != null 
        ? -((_valuationResult['engine_deduction_lkr'] as num) / (_fairValueMillion * 1000000) * 100).round()
        : -((100 - _engineConditionScore) / 5).round();

    // Estimate mileage and year depreciation
    int mileageKm = args['mileage_km'] ?? 80000;
    int mileageDep = -(mileageKm / 10000).round();
    
    int year = args['maf_year'] ?? 2015;
    int yearDep = -(2026 - year);

    final factors = [
      {'label': 'Body Condition', 'value': bodyDep, 'color': _getScoreColor(_bodyConditionScore)},
      {'label': 'Mileage', 'value': mileageDep, 'color': AppColors.statusAmber},
      {'label': 'Mechanical', 'value': mechDep, 'color': _getScoreColor(_engineConditionScore)},
      {'label': 'Year', 'value': yearDep, 'color': AppColors.primaryBlue},
    ];


    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2035),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Depreciation Factors",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
          ),
          const SizedBox(height: 14),
          ...factors.map((f) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  children: [
                    SizedBox(
                      width: 110,
                      child: Text(
                        f['label'] as String,
                        style: const TextStyle(color: Colors.white, fontSize: 13),
                      ),
                    ),
                    Expanded(
                      child: Stack(
                        children: [
                          Container(
                            height: 8,
                            decoration: BoxDecoration(
                              color: Colors.grey.shade800,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                          FractionallySizedBox(
                            widthFactor: (f['value'] as int).abs() / 20.0,
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
        valueColor = (item['display'] == 'Original' || (item['display']?.toString().contains('%') ?? false) && (double.tryParse(item['display']?.toString().replaceAll('%', '') ?? '0') ?? 0) >= 80) 
            ? AppColors.statusGreen 
            : (item['display'] == 'Needs Review' ? AppColors.statusAmber : AppColors.statusRed);
      } else {
        double score = item['score'] as double;
        if (score >= 80) {
          valueColor = AppColors.statusGreen;
        } else if (score >= 50) {
          valueColor = AppColors.statusAmber;
        } else {
          valueColor = AppColors.statusRed;
        }
      }

      return Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF1A2035),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              item['label'] as String,
              style: const TextStyle(color: Colors.white, fontSize: 14),
            ),
            Text(
              item['display'] as String,
              style: TextStyle(color: valueColor, fontWeight: FontWeight.bold, fontSize: 14),
            ),
          ],
        ),
      );
    }).toList();
  }

  Color _getScoreColor(double score) {
    if (score >= 80) return AppColors.statusGreen;
    if (score >= 50) return AppColors.statusAmber;
    return AppColors.statusRed;
  }
}

