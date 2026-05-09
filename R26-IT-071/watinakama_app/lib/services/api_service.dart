import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiConfig {
  // Change this to your PC's IP address when using real device
  // Use 10.0.2.2 for Android emulator
  static const String baseIp = '10.0.2.2';

  static const String vinApi = 'http://$baseIp:8000';
  static const String bodyApi = 'http://$baseIp:8080';
  static const String engineApi = 'http://$baseIp:5003';
  static const String valuationApi = 'http://$baseIp:5004';

  static const String engineToken =
      'Bearer 097ad29076b1a4d2121e9ee67f478357e3d883ebd57d3ab609ae725495f79bcf';
  static const String valuationToken = 'Bearer watinakama-valuation-api-2026';
}

class ApiService {
  /// Analyzes engine sound from an audio file.
  Future<Map<String, dynamic>> analyzeEngine(File audioFile) async {
    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('${ApiConfig.engineApi}/api/v1/analyze'));
      
      request.headers['Authorization'] = ApiConfig.engineToken;
      request.files.add(
        await http.MultipartFile.fromPath('audio_file', audioFile.path),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Analyzes vehicle body condition from a list of 5 images.
  /// Field names: front, rear, left, right, roof
  Future<Map<String, dynamic>> analyzeBody(List<File> images) async {
    try {
      var request =
          http.MultipartRequest('POST', Uri.parse('${ApiConfig.bodyApi}/analyze'));

      final List<String> fieldNames = ['front', 'rear', 'left', 'right', 'roof'];

      for (int i = 0; i < images.length && i < fieldNames.length; i++) {
        request.files.add(
          await http.MultipartFile.fromPath(fieldNames[i], images[i].path),
        );
      }

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Scans VIN from an image file.
  Future<Map<String, dynamic>> scanVin(File vinImage) async {
    try {
      var request =
          http.MultipartRequest('POST', Uri.parse('${ApiConfig.vinApi}/predict'));
      
      request.files.add(
        await http.MultipartFile.fromPath('file', vinImage.path),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Gets the valuation for the vehicle based on the provided data.
  Future<Map<String, dynamic>> getValuation(Map<String, dynamic> data) async {
    try {
      var response = await http.post(
        Uri.parse('${ApiConfig.valuationApi}/api/v1/valuate'),
        headers: {
          'Authorization': ApiConfig.valuationToken,
          'Content-Type': 'application/json',
        },
        body: jsonEncode(data),
      );

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Verifies if a backend service is running.
  /// Checks $apiUrl/health and $apiUrl/api/v1/health.
  Future<bool> checkHealth(String apiUrl) async {
    try {
      // Try /health first
      var response = await http.get(Uri.parse('$apiUrl/health')).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) return true;

      // Try /api/v1/health if /health fails
      response = await http.get(Uri.parse('$apiUrl/api/v1/health')).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
