import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_text_field.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';
import '../widgets/custom_toast.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final TextEditingController _emailController = TextEditingController();
  final AuthService _auth = AuthService();
  bool _isLoading = false;

  Future<void> _handleForgotPassword() async {
    final email = _emailController.text.trim();

    if (email.isEmpty) {
      _showError('Please enter your email address');
      return;
    }

    setState(() => _isLoading = true);

    final result = await _auth.forgotPassword(email);

    if (mounted) {
      setState(() => _isLoading = false);

      if (result['status'] == 'success') {
        ToastService.show(context, 'Reset email sent! Check your inbox and follow the link to reset your password.');
        // Wait a moment so the user can read the toast, then navigate back
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) Navigator.pushReplacementNamed(context, '/login');
        });
      } else {
        _showError(result['message']);
      }
    }
  }

  void _showError(String message) {
    ToastService.show(context, message, isError: true);
  }

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      body: Column(
        children: [
          const WaveHeader(height: 260),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  const Text(
                    "Forgot Password",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: AppColors.textWhite,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "Enter your email and we will send you a password reset link",
                    style: TextStyle(
                      fontSize: 13,
                      color: AppColors.textGray,
                    ),
                  ),
                  const SizedBox(height: 32),
                  
                  CustomTextField(
                    label: "Email Address",
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    hint: "example@mail.com",
                  ),
                  
                  const SizedBox(height: 32),
                  
                  CustomButton(
                    text: "Send Reset Email",
                    isLoading: _isLoading,
                    onPressed: _handleForgotPassword,
                  ),
                  
                  const SizedBox(height: 24),
                  
                  Center(
                    child: TextButton(
                      onPressed: () => Navigator.pop(context),
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
