import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../utils/error_messages.dart';

class ApiConfig {
  // The gateway is the single public entry point; it proxies to the 4
  // backend services and injects their auth tokens server-side.
  static const String gatewayBaseUrl = 'https://watinakama-gateway.azurewebsites.net';

  static const String vinApi = '$gatewayBaseUrl/vin';
  static const String bodyApi = '$gatewayBaseUrl/body';
  static const String engineApi = '$gatewayBaseUrl/engine';
  static const String valuationApi = '$gatewayBaseUrl/valuation';
}

class ApiService {
  /// Analyzes engine sound from one or more audio stages.
  /// Keys in phasePaths: 'start', 'idle', 'acceleration'
  Future<Map<String, dynamic>> analyzeEngine(Map<String, String> phasePaths) async {
    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('${ApiConfig.engineApi}/api/v1/analyze'));

      // Map the keys from the recording screen to the backend expected field names
      if (phasePaths.containsKey('engine_start')) {
        request.files.add(await http.MultipartFile.fromPath('audio_start', phasePaths['engine_start']!));
      } else if (phasePaths.containsKey('start')) {
        // Backward compatibility for different screen implementations
        request.files.add(await http.MultipartFile.fromPath('audio_start', phasePaths['start']!));
      }
      
      if (phasePaths.containsKey('idle')) {
        request.files.add(await http.MultipartFile.fromPath('audio_idle', phasePaths['idle']!));
      }
      
      if (phasePaths.containsKey('acceleration')) {
        request.files.add(await http.MultipartFile.fromPath('audio_acceleration', phasePaths['acceleration']!));
      }

      // Backward compatibility for single file/old backend
      if (request.files.isEmpty && phasePaths.isNotEmpty) {
        // If no multi-stage keys matched but we have paths, send the first one as audio_file
        request.files.add(await http.MultipartFile.fromPath('audio_file', phasePaths.values.first));
      }

      if (request.files.isEmpty) return {"status": "error", "message": "No audio files provided"};

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        String msg = "API returned status code ${response.statusCode}";
        try {
          final errBody = jsonDecode(response.body);
          msg = errBody['detail'] ?? errBody['message'] ?? errBody['error'] ?? msg;
        } catch (_) {}
        return {"status": "error", "status_code": response.statusCode, "message": msg};
      }

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e, fallback: "Couldn't reach the server. Please try again.")};
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

      if (response.statusCode != 200) {
        String msg = "API returned status code ${response.statusCode}";
        try {
          final errBody = jsonDecode(response.body);
          msg = errBody['detail'] ?? errBody['message'] ?? msg;
        } catch (_) {}
        return {"status": "error", "status_code": response.statusCode, "message": msg};
      }

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e, fallback: "Couldn't reach the server. Please try again.")};
    }
  }

  /// Validates a single vehicle image against the expected view.
  /// [imagePath] – path to the image file
  /// [expectedView] – one of: front, rear, left, right, roof
  ///
  /// Returns a map with:
  ///   valid      – bool
  ///   correct    – bool
  ///   predicted  – String (e.g. "Left")
  ///   expected   – String (e.g. "Left")
  ///   confidence – double
  ///   message    – String (human-readable verdict)
  Future<Map<String, dynamic>> validateView(String imagePath, String expectedView) async {
    try {
      final uri = Uri.parse(
        '${ApiConfig.bodyApi}/api/v1/validate-view?expected_view=$expectedView',
      );
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('image', imagePath));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        String msg = "View validation API returned status ${response.statusCode}";
        try {
          final errBody = jsonDecode(response.body);
          msg = errBody['detail'] ?? errBody['message'] ?? msg;
        } catch (_) {}
        // On API error, allow through (fail-open)
        return {"valid": true, "correct": true, "status": "error", "message": msg};
      }

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      // Network error – fail-open so users aren't blocked
      return {"valid": true, "correct": true, "status": "error", "message": e.toString()};
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

      if (response.statusCode != 200) {
        String msg = "API returned status code ${response.statusCode}";
        try {
          final errBody = jsonDecode(response.body);
          msg = errBody['detail'] ?? errBody['message'] ?? msg;
        } catch (_) {}
        return {"status": "error", "status_code": response.statusCode, "message": msg};
      }

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e, fallback: "Couldn't reach the server. Please try again.")};
    }
  }

  /// Gets the valuation for the vehicle based on the provided data.
  Future<Map<String, dynamic>> getValuation(Map<String, dynamic> data) async {
    try {
      var response = await http.post(
        Uri.parse('${ApiConfig.valuationApi}/api/v1/valuate'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(data),
      );

      if (response.statusCode != 200) {
        String msg = "API returned status code ${response.statusCode}";
        try {
          final errBody = jsonDecode(response.body);
          msg = errBody['detail'] ?? errBody['message'] ?? errBody['error'] ?? msg;
        } catch (_) {}
        return {"status": "error", "status_code": response.statusCode, "message": msg};
      }

      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e, fallback: "Couldn't reach the server. Please try again.")};
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
