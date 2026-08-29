import 'package:shared_preferences/shared_preferences.dart';
import '../models/inspection_record.dart';

/// Persists completed inspection summaries locally so the home screen can
/// list them for later reference.
class InspectionHistoryService {
  static const _storageKey = 'inspection_history';
  static const _maxRecords = 20;

  Future<SharedPreferences> get _prefs async => await SharedPreferences.getInstance();

  /// Newest-first list of saved inspections.
  Future<List<InspectionRecord>> getHistory() async {
    final prefs = await _prefs;
    final raw = prefs.getString(_storageKey);
    if (raw == null || raw.isEmpty) return [];

    try {
      final records = InspectionRecord.decodeList(raw);
      records.sort((a, b) => b.date.compareTo(a.date));
      return records;
    } catch (_) {
      return [];
    }
  }

  Future<void> addInspection(InspectionRecord record) async {
    final prefs = await _prefs;
    final existing = await getHistory();
    existing.insert(0, record);
    if (existing.length > _maxRecords) {
      existing.removeRange(_maxRecords, existing.length);
    }
    await prefs.setString(_storageKey, InspectionRecord.encodeList(existing));
  }

  Future<void> clearHistory() async {
    final prefs = await _prefs;
    await prefs.remove(_storageKey);
  }
}
