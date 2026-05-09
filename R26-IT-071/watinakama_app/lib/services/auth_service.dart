import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FlutterSecureStorage _secureStorage = const FlutterSecureStorage(
    aOptions: AndroidOptions(
      encryptedSharedPreferences: true,
    ),
  );

  // Future to get shared preferences
  Future<SharedPreferences> get _prefs async => await SharedPreferences.getInstance();

  /// Register a new user
  Future<Map<String, dynamic>> register(
      String fullName, String email, String password, String confirmPassword) async {
    
    // Manual validations
    if (password != confirmPassword) {
      return {"status": "error", "message": "Passwords do not match"};
    }
    if (password.length < 8) {
      return {"status": "error", "message": "Password must be at least 8 characters"};
    }

    try {
      UserCredential credential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      // Update display name
      await credential.user?.updateDisplayName(fullName);
      
      // Save credentials for biometric login
      await _secureStorage.write(key: 'email', value: email);
      await _secureStorage.write(key: 'password', value: password);

      // Save to SharedPreferences
      final prefs = await _prefs;
      await prefs.setString('user_name', fullName);

      return {
        "status": "success",
        "message": "Account created successfully",
      };
    } on FirebaseAuthException catch (e) {
      String friendlyMessage = "Registration failed. Please try again.";
      
      if (e.code == 'email-already-in-use') {
        friendlyMessage = "This email is already registered";
      } else if (e.code == 'weak-password') {
        friendlyMessage = "Password must be at least 6 characters";
      } else if (e.code == 'invalid-email') {
        friendlyMessage = "Please enter a valid email address";
      }

      return {"status": "error", "message": friendlyMessage};
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Login an existing user
  Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      UserCredential credential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      final String displayName = credential.user?.displayName ?? "User";
      final String userEmail = credential.user?.email ?? email;

      // Save credentials for biometric login
      await _secureStorage.write(key: 'email', value: userEmail);
      await _secureStorage.write(key: 'password', value: password);

      // Save to SharedPreferences
      final prefs = await _prefs;
      await prefs.setString('user_name', displayName);
      await prefs.setString('user_email', userEmail);

      return {
        "status": "success",
        "message": "Login successful",
        "user_name": displayName,
      };
    } on FirebaseAuthException catch (e) {
      String friendlyMessage = "Login failed. Please check your credentials.";

      if (e.code == 'user-not-found') {
        friendlyMessage = "No account found with this email";
      } else if (e.code == 'wrong-password') {
        friendlyMessage = "Incorrect password";
      } else if (e.code == 'invalid-credential') {
        friendlyMessage = "Invalid email or password";
      }

      return {"status": "error", "message": friendlyMessage};
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Send password reset email
  Future<Map<String, dynamic>> forgotPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
      return {
        "status": "success",
        "message": "Password reset email sent. Check your inbox.",
      };
    } on FirebaseAuthException catch (e) {
      String friendlyMessage = "Failed to send reset email.";

      if (e.code == 'user-not-found') {
        friendlyMessage = "No account found with this email";
      }

      return {"status": "error", "message": friendlyMessage};
    } catch (e) {
      return {"status": "error", "message": e.toString()};
    }
  }

  /// Logout user
  Future<void> logout() async {
    await _auth.signOut();
    
    final prefs = await _prefs;
    final bool bioEnabled = prefs.getBool('biometric_enabled') ?? false;
    
    // Clear secured credentials only if biometric login is NOT enabled
    // This allows users to log back in using biometrics
    if (!bioEnabled) {
      await _secureStorage.deleteAll();
    }

    await prefs.remove('user_name');
    await prefs.remove('user_email');
  }

  /// Biometric Login using saved credentials
  Future<Map<String, dynamic>> loginWithBiometrics() async {
    try {
      final String? email = await _secureStorage.read(key: 'email');
      final String? password = await _secureStorage.read(key: 'password');

      if (email != null && password != null) {
        return await login(email, password);
      }

      return {
        "status": "error",
        "message": "No credentials stored. Please login with email first.",
      };
    } catch (e) {
      return {
        "status": "error",
        "message": "Failed to read credentials: ${e.toString()}",
      };
    }
  }

  /// Check if user is logged in
  bool isLoggedIn() {
    return _auth.currentUser != null;
  }

  /// Get current Firebase user
  User? getCurrentUser() {
    return _auth.currentUser;
  }

  /// Get user's full name
  Future<String> getUserName() async {
    // 1. Try Firebase Display Name
    if (_auth.currentUser != null) {
      // If display name is missing, try to reload the user to fetch the latest profile
      if (_auth.currentUser!.displayName == null) {
        await _auth.currentUser!.reload();
      }
      
      if (_auth.currentUser!.displayName != null) {
        return _auth.currentUser!.displayName!;
      }
    }

    // 2. Try SharedPreferences
    final prefs = await _prefs;
    final String? localName = prefs.getString('user_name');
    if (localName != null) {
      return localName;
    }

    return "User";
  }

  /// Get user's profile picture path
  Future<String?> getProfilePicPath() async {
    final prefs = await _prefs;
    return prefs.getString('user_profile_pic');
  }
}
