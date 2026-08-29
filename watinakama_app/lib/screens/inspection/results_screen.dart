import 'package:flutter/material.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../services/api_service.dart';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  Map<String, dynamic> _vehicleData = {};
  // ignore: unused_field
  final Map<String, dynamic> _valuationResult = {};
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
        _engineConditionScore = (args['confidence'] ?? 0).toDouble() * 100; // placeholder for engine score
        _vinCondition = (args['vin_status'] == 'original') ? 'Original' : 'Unknown';
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
        _engineConditionScore = (args['mhs_score'] ?? 100).toDouble();
        _vinCondition = args['vin_status'] ?? 'original';
        _verdict = result['verdict'] ?? 'FAIR_PRICE';
        _explanation = result['explanation'] ?? '';
        _isLoading = false;
      });
    } else {
      // Use demo data if API fails
      setState(() {
        _fairValueMillion = 3.85;
        _rangeMinMillion = 3.6;
        _rangeMaxMillion = 4.1;
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
    final factors = [
      {'label': 'Body Condition', 'value': -8, 'color': AppColors.statusGreen},
      {'label': 'Mileage', 'value': -12, 'color': AppColors.statusAmber},
      {'label': 'Mechanical', 'value': -9, 'color': AppColors.statusAmber},
      {'label': 'Year', 'value': -6, 'color': AppColors.primaryBlue},
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
        valueColor = (item['display'] == 'Original') ? AppColors.statusGreen : AppColors.statusRed;
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
}
