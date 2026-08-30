import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';
import '../widgets/custom_toast.dart';
import '../utils/error_messages.dart';

class EmailVerificationScreen extends StatefulWidget {
  final String email;
  final String password;

  const EmailVerificationScreen({
    super.key,
    required this.email,
    required this.password,
  });

  @override
  State<EmailVerificationScreen> createState() => _EmailVerificationScreenState();
}

class _EmailVerificationScreenState extends State<EmailVerificationScreen> {
  final AuthService _auth = AuthService();
  bool _isChecking = false;
  bool _isResending = false;

  Future<void> _handleContinue() async {
    setState(() => _isChecking = true);

    // Reuses login() so secure-storage credentials and prefs are populated
    // the same way a normal login would, then routes home on success.
    final result = await _auth.login(widget.email, widget.password);

    if (mounted) {
      setState(() => _isChecking = false);

      if (result['status'] == 'success') {
        Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
      } else if (result['code'] == 'email-not-verified') {
        ToastService.show(
          context,
          "Email not verified yet. Please check your inbox and tap the link.",
          isError: true,
        );
      } else {
        ToastService.show(context, result['message'], isError: true);
      }
    }
  }

  Future<void> _handleResend() async {
    setState(() => _isResending = true);

    // If still signed in from registration, resend directly; otherwise
    // fall back to a brief sign-in/out cycle.
    final currentUser = _auth.getCurrentUser();
    Map<String, dynamic> result;
    if (currentUser != null && currentUser.email == widget.email) {
      try {
        await currentUser.sendEmailVerification();
        result = {"status": "success", "message": "Verification email sent. Check your inbox."};
      } catch (e) {
        result = {"status": "error", "message": friendlyErrorMessage(e)};
      }
    } else {
      result = await _auth.resendVerificationEmail(widget.email, widget.password);
    }

    if (mounted) {
      setState(() => _isResending = false);
      ToastService.show(context, result['message'], isError: result['status'] != 'success');
    }
  }

  void _backToLogin() async {
    await _auth.logout();
    if (mounted) {
      Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      body: Column(
        children: [
          const WaveHeader(height: 240),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  const Icon(Icons.mark_email_read_outlined, size: 56, color: AppColors.primaryBlue),
                  const SizedBox(height: 16),
                  const Text(
                    "Verify Your Email",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: AppColors.textWhite,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    "We sent a verification link to ${widget.email}. "
                    "Open it, then tap Continue below.",
                    style: const TextStyle(
                      fontSize: 13,
                      color: AppColors.textGray,
                    ),
                  ),
                  const SizedBox(height: 32),

                  CustomButton(
                    text: "I've Verified, Continue",
                    isLoading: _isChecking,
                    onPressed: _handleContinue,
                  ),
                  const SizedBox(height: 12),
                  CustomButton(
                    text: "Resend Email",
                    variant: AppButtonVariant.outline,
                    isLoading: _isResending,
                    onPressed: _handleResend,
                  ),

                  const SizedBox(height: 24),
                  Center(
                    child: TextButton(
                      onPressed: _backToLogin,
                      child: const Text(
                        "Back to Login",
                        style: TextStyle(
                          color: AppColors.linkBlue,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
