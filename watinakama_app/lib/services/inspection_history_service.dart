import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/inspection_record.dart';

/// Stores completed inspection summaries in Cloud Firestore under the signed-in
/// user, at users/{uid}/inspections/{id}, so they follow the account across
/// devices and reinstalls. Firestore's offline cache keeps the history readable
/// and queues new reports when the phone has no connection.
///
/// Only the result summary is stored, never audio or photos.
class InspectionHistoryService {
  /// Key used by older app versions, which kept history on the device only.
  static const _legacyStorageKey = 'inspection_history';
  static const _maxRecords = 20;

  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  CollectionReference<Map<String, dynamic>>? get _collection {
    final uid = _auth.currentUser?.uid;
    if (uid == null) return null;
    return _db.collection('users').doc(uid).collection('inspections');
  }

  /// Newest-first list of saved inspections for the signed-in user.
  Future<List<InspectionRecord>> getHistory() async {
    final collection = _collection;
    if (collection == null) return [];

    try {
      await _migrateLocalHistory(collection);
      final snapshot = await collection.orderBy('date', descending: true).limit(_maxRecords).get();
      return snapshot.docs.map(_fromDocument).whereType<InspectionRecord>().toList();
    } catch (e) {
      debugPrint('Could not load inspection history: $e');
      return [];
    }
  }

  Future<void> addInspection(InspectionRecord record) async {
    final collection = _collection;
    if (collection == null) return;

    try {
      // Without awaiting the server, the write still lands in the offline cache
      // straight away and is sent when the phone is back online.
      collection.doc(record.id).set(_toDocument(record));
    } catch (e) {
      debugPrint('Could not save inspection: $e');
    }
  }

  Future<void> clearHistory() async {
    final collection = _collection;
    if (collection == null) return;

    try {
      final snapshot = await collection.get();
      final batch = _db.batch();
      for (final doc in snapshot.docs) {
        batch.delete(doc.reference);
      }
      await batch.commit();
    } catch (e) {
      debugPrint('Could not clear inspection history: $e');
    }
  }

  /// Moves reports saved by older app versions into the user's Firestore
  /// history once, then removes the device-only copy.
  Future<void> _migrateLocalHistory(CollectionReference<Map<String, dynamic>> collection) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_legacyStorageKey);
    if (raw == null || raw.isEmpty) return;

    try {
      final batch = _db.batch();
      for (final record in InspectionRecord.decodeList(raw)) {
        batch.set(collection.doc(record.id), _toDocument(record));
      }
      await batch.commit();
    } catch (e) {
      debugPrint('Could not migrate local inspection history: $e');
      return; // keep the local copy and try again next time
    }
    await prefs.remove(_legacyStorageKey);
  }

  Map<String, dynamic> _toDocument(InspectionRecord record) {
    final data = record.toJson();
    data['date'] = Timestamp.fromDate(record.date); // a Timestamp sorts correctly in queries
    return data;
  }

  InspectionRecord? _fromDocument(QueryDocumentSnapshot<Map<String, dynamic>> doc) {
    try {
      final data = Map<String, dynamic>.from(doc.data());
      final date = data['date'];
      if (date is Timestamp) data['date'] = date.toDate().toIso8601String();
      data['id'] ??= doc.id;
      return InspectionRecord.fromJson(data);
    } catch (e) {
      debugPrint('Skipping unreadable inspection ${doc.id}: $e');
      return null;
    }
  }
}
