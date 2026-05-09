import 'package:local_auth/local_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';

class BiometricService {
  final LocalAuthentication _auth = LocalAuthentication();

  // Future to get shared preferences
  Future<SharedPreferences> get _prefs async => await SharedPreferences.getInstance();

  /// Check if biometric authentication is available on the device
  Future<bool> isAvailable() async {
    try {
      final bool canCheck = await _auth.canCheckBiometrics;
      final bool isSupported = await _auth.isDeviceSupported();
      return canCheck && isSupported;
    } catch (e) {
      return false;
    }
  }

  /// Perform biometric authentication
  Future<bool> authenticate() async {
    try {
      return await _auth.authenticate(
        localizedReason: 'Verify your identity to login to Watinakama.lk',
        options: const AuthenticationOptions(
          biometricOnly: false,
          stickyAuth: true,
        ),
      );
    } catch (e) {
      return false;
    }
  }

  /// Enable or disable biometric login preference
  Future<void> setEnabled(bool value) async {
    final prefs = await _prefs;
    await prefs.setBool('biometric_enabled', value);
  }

  /// Check if biometric login is enabled in preferences
  Future<bool> isEnabled() async {
    final prefs = await _prefs;
    return prefs.getBool('biometric_enabled') ?? false;
  }
}
