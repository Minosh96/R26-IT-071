import 'dart:convert';

/// A saved summary of a completed vehicle inspection, kept locally so the
/// user can look back at past results from the home screen.
class InspectionRecord {
  final String id;
  final DateTime date;
  final String make;
  final String model;
  final double fairValueMillion;
  final double rangeMinMillion;
  final double rangeMaxMillion;
  final String verdict;
  final double bodyConditionScore;
  final double engineConditionScore;
  final String vinCondition;

  InspectionRecord({
    required this.id,
    required this.date,
    required this.make,
    required this.model,
    required this.fairValueMillion,
    required this.rangeMinMillion,
    required this.rangeMaxMillion,
    required this.verdict,
    required this.bodyConditionScore,
    required this.engineConditionScore,
    required this.vinCondition,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'date': date.toIso8601String(),
        'make': make,
        'model': model,
        'fair_value_million': fairValueMillion,
        'range_min_million': rangeMinMillion,
        'range_max_million': rangeMaxMillion,
        'verdict': verdict,
        'body_condition_score': bodyConditionScore,
        'engine_condition_score': engineConditionScore,
        'vin_condition': vinCondition,
      };

  factory InspectionRecord.fromJson(Map<String, dynamic> json) {
    return InspectionRecord(
      id: json['id'] as String,
      date: DateTime.parse(json['date'] as String),
      make: json['make'] as String? ?? '',
      model: json['model'] as String? ?? '',
      fairValueMillion: (json['fair_value_million'] as num?)?.toDouble() ?? 0,
      rangeMinMillion: (json['range_min_million'] as num?)?.toDouble() ?? 0,
      rangeMaxMillion: (json['range_max_million'] as num?)?.toDouble() ?? 0,
      verdict: json['verdict'] as String? ?? '',
      bodyConditionScore: (json['body_condition_score'] as num?)?.toDouble() ?? 0,
      engineConditionScore: (json['engine_condition_score'] as num?)?.toDouble() ?? 0,
      vinCondition: json['vin_condition'] as String? ?? 'Unknown',
    );
  }

  static String encodeList(List<InspectionRecord> records) =>
      jsonEncode(records.map((r) => r.toJson()).toList());

  static List<InspectionRecord> decodeList(String raw) {
    final decoded = jsonDecode(raw) as List;
    return decoded
        .map((e) => InspectionRecord.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
