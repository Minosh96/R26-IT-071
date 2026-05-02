import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_text_field.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _passController = TextEditingController();
  final TextEditingController _confirmController = TextEditingController();
  
  final AuthService _auth = AuthService();
  bool _isLoading = false;

  Future<void> _handleRegister() async {
    final name = _nameController.text.trim();
    final email = _emailController.text.trim();
    final phone = _phoneController.text.trim();
    final password = _passController.text.trim();
    final confirmPassword = _confirmController.text.trim();

    // Validations
    if (name.isEmpty || email.isEmpty || phone.isEmpty || password.isEmpty || confirmPassword.isEmpty) {
      _showError('All fields are required');
      return;
    }

    if (password != confirmPassword) {
      _showError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      _showError('Password must be at least 8 characters');
      return;
    }

    setState(() => _isLoading = true);

    final result = await _auth.register(name, email, password, confirmPassword);

    if (mounted) {
      setState(() => _isLoading = false);

      if (result['status'] == 'success') {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Account created! Please login.'),
            backgroundColor: AppColors.statusGreen,
          ),
        );
        Navigator.pushReplacementNamed(context, '/login');
      } else {
        _showError(result['message']);
      }
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.redAccent,
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _passController.dispose();
    _confirmController.dispose();
    super.dispose();
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
                  const Text(
                    "Create Account",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    "Join your professional network today",
                    style: TextStyle(
                      fontSize: 12,
                      color: AppColors.textGray,
                    ),
                  ),
                  const SizedBox(height: 16),
                  
                  CustomTextField(
                    label: "Full Name",
                    controller: _nameController,
                    hint: "John Doe",
                  ),
                  const SizedBox(height: 10),
                  
                  CustomTextField(
                    label: "E-mail",
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    hint: "example@mail.com",
                  ),
                  const SizedBox(height: 10),
                  
                  const Text(
                    "Phone Number",
                    style: TextStyle(
                      color: AppColors.textGray,
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Container(
                        width: 70,
                        height: 44,
                        decoration: BoxDecoration(
                          color: AppColors.darkNavyCard,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: AppColors.textFieldBorder),
                        ),
                        child: const Center(
                          child: Text(
                            "+94",
                            style: TextStyle(color: Colors.white, fontSize: 14),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: CustomTextField(
                          label: "",
                          controller: _phoneController,
                          keyboardType: TextInputType.phone,
                          hint: "7X XXX XXXX",
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  
                  Row(
                    children: [
                      Expanded(
                        child: CustomTextField(
                          label: "Password",
                          controller: _passController,
                          isPassword: true,
                          hint: "••••••",
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: CustomTextField(
                          label: "Confirm Password",
                          controller: _confirmController,
                          isPassword: true,
                          hint: "••••••",
                        ),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  
                  Center(
                    child: RichText(
                      text: TextSpan(
                        style: const TextStyle(fontSize: 13, color: AppColors.textGray),
                        children: [
                          const TextSpan(text: "Already have an "),
                          TextSpan(
                            text: "account?",
                            style: const TextStyle(color: AppColors.linkBlue),
                            recognizer: TapGestureRecognizer()
                              ..onTap = () => Navigator.pushReplacementNamed(context, '/login'),
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  CustomButton(
                    text: "Register",
                    isLoading: _isLoading,
                    onPressed: _handleRegister,
                  ),
                  
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
