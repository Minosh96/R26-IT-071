import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../utils/error_messages.dart';

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

      // Send verification email; the account stays unverified until the
      // user opens the link, and login() blocks unverified accounts.
      await credential.user?.sendEmailVerification();

      // Save credentials for biometric login
      await _secureStorage.write(key: 'email', value: email);
      await _secureStorage.write(key: 'password', value: password);

      // Save to SharedPreferences
      final prefs = await _prefs;
      await prefs.setString('user_name', fullName);

      return {
        "status": "success",
        "message": "Account created. Check your email to verify your address before logging in.",
      };
    } on FirebaseAuthException catch (e) {
      return {"status": "error", "message": firebaseAuthErrorMessage(e)};
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e)};
    }
  }

  /// Login an existing user
  Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      UserCredential credential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      // Refresh the cached user so emailVerified reflects the latest state.
      await credential.user?.reload();
      final User? refreshedUser = _auth.currentUser;

      if (refreshedUser != null && !refreshedUser.emailVerified) {
        await _auth.signOut();
        return {
          "status": "error",
          "code": "email-not-verified",
          "message": "Please verify your email before logging in. Check your inbox for the verification link.",
        };
      }

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
      return {"status": "error", "message": firebaseAuthErrorMessage(e)};
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e)};
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
      return {"status": "error", "message": firebaseAuthErrorMessage(e)};
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e)};
    }
  }

  /// Resend the verification email for an account that hasn't logged in yet.
  /// Signs in briefly (login() already signs unverified users back out),
  /// re-sends the email if still unverified, then signs out again.
  Future<Map<String, dynamic>> resendVerificationEmail(String email, String password) async {
    try {
      UserCredential credential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      await credential.user?.reload();
      final User? user = _auth.currentUser;

      if (user == null) {
        return {"status": "error", "message": "Could not verify account. Please try again."};
      }

      if (user.emailVerified) {
        await _auth.signOut();
        return {"status": "error", "message": "This email is already verified. You can log in."};
      }

      await user.sendEmailVerification();
      await _auth.signOut();

      return {
        "status": "success",
        "message": "Verification email sent. Check your inbox.",
      };
    } on FirebaseAuthException catch (e) {
      return {"status": "error", "message": firebaseAuthErrorMessage(e)};
    } catch (e) {
      return {"status": "error", "message": friendlyErrorMessage(e)};
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
        "message": "Couldn't sign in with biometrics. Please login with your password.",
      };
    }
  }

  /// Check if user is logged in with a verified email.
  /// Reloads the cached user first so a verification done outside the app
  /// (e.g. clicking the email link) is picked up on the next app open.
  Future<bool> isLoggedIn() async {
    final User? user = _auth.currentUser;
    if (user == null) return false;

    try {
      await user.reload();
    } catch (_) {
      // Offline or transient error: fall back to the last known state.
    }

    return _auth.currentUser?.emailVerified ?? false;
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
