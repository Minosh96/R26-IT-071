import 'dart:async';
import 'dart:io';
import 'package:firebase_auth/firebase_auth.dart';

/// Converts a caught error into a short, plain-language message safe to
/// show in a toast. Never surfaces raw exception text, stack traces, or
/// class names to end users.
String friendlyErrorMessage(
  Object error, {
  String fallback = "Something went wrong. Please try again.",
}) {
  if (error is FirebaseAuthException) {
    return firebaseAuthErrorMessage(error);
  }
  if (error is SocketException || error is HttpException) {
    return "No internet connection. Please check your network and try again.";
  }
  if (error is TimeoutException) {
    return "The request timed out. Please try again.";
  }
  return fallback;
}

/// Maps Firebase Auth error codes (shared by register/login/reset/password
/// change) to plain-language messages.
String firebaseAuthErrorMessage(FirebaseAuthException e) {
  switch (e.code) {
    case 'email-already-in-use':
      return "This email is already registered";
    case 'weak-password':
      return "Password must be at least 8 characters";
    case 'invalid-email':
      return "Please enter a valid email address";
    case 'user-not-found':
      return "No account found with this email";
    case 'wrong-password':
    case 'invalid-credential':
      return "Incorrect email or password";
    case 'too-many-requests':
      return "Too many attempts. Please wait a moment and try again.";
    case 'network-request-failed':
      return "No internet connection. Please check your network and try again.";
    case 'requires-recent-login':
      return "Please logout and login again before changing your password";
    case 'user-disabled':
      return "This account has been disabled. Please contact support.";
    default:
      return "Something went wrong. Please try again.";
  }
}
