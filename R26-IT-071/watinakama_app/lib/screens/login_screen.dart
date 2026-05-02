import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_text_field.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';
import '../services/biometric_service.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  
  final AuthService _auth = AuthService();
  final BiometricService _bio = BiometricService();
  
  bool _isLoading = false;
  bool _bioAvailable = false;
  bool _bioEnabled = false;

  @override
  void initState() {
    super.initState();
    _checkBiometric();
  }

  Future<void> _checkBiometric() async {
    _bioAvailable = await _bio.isAvailable();
    _bioEnabled = await _bio.isEnabled();
    if (mounted) setState(() {});
  }

  Future<void> _handleLogin() async {
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    if (email.isEmpty || password.isEmpty) {
      _showError('Please enter both email and password');
      return;
    }

    setState(() => _isLoading = true);

    final result = await _auth.login(email, password);

    if (mounted) {
      setState(() => _isLoading = false);

      if (result['status'] == 'success') {
        // If biometric available but not enabled, ask the user
        if (_bioAvailable && !_bioEnabled) {
          await _showBiometricDialog();
        }
        Navigator.pushReplacementNamed(context, '/home');
      } else {
        _showError(result['message']);
      }
    }
  }

  Future<void> _handleBiometricLogin() async {
    if (!_bioEnabled) {
      _showError("Login with email first to enable biometric");
      return;
    }

    final authenticated = await _bio.authenticate();
    
    if (authenticated) {
      setState(() => _isLoading = true);
      
      final result = await _auth.loginWithBiometrics();
      
      if (mounted) {
        setState(() => _isLoading = false);
        
        if (result['status'] == 'success') {
          Navigator.pushReplacementNamed(context, '/home');
        } else {
          _showError(result['message']);
        }
      }
    } else {
      _showError("Biometric failed. Try email login.");
    }
  }

  Future<void> _showBiometricDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.darkNavySurface,
        title: const Text("Enable Biometrics", style: TextStyle(color: Colors.white)),
        content: const Text(
          "Would you like to enable fingerprint/face login for future sessions?",
          style: TextStyle(color: AppColors.textGray),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("No", style: TextStyle(color: Colors.white70)),
          ),
          TextButton(
            onPressed: () async {
              await _bio.setEnabled(true);
              if (mounted) Navigator.pop(context);
            },
            child: const Text("Yes", style: TextStyle(color: AppColors.primaryBlue)),
          ),
        ],
      ),
    );
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
    _emailController.dispose();
    _passwordController.dispose();
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
                    "Welcome Back",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    "Sign in to your professional account",
                    style: TextStyle(
                      fontSize: 12,
                      color: AppColors.textGray,
                    ),
                  ),
                  const SizedBox(height: 20),
                  
                  CustomTextField(
                    label: "Email",
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    hint: "example@mail.com",
                  ),
                  const SizedBox(height: 12),
                  CustomTextField(
                    label: "Password",
                    controller: _passwordController,
                    isPassword: true,
                    hint: "••••••••",
                  ),
                  const SizedBox(height: 12),
                  
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      RichText(
                        text: TextSpan(
                          style: const TextStyle(fontSize: 13, color: AppColors.textGray),
                          children: [
                            const TextSpan(text: "Don't have an "),
                            TextSpan(
                              text: "account?",
                              style: const TextStyle(color: AppColors.linkBlue),
                              recognizer: TapGestureRecognizer()
                                ..onTap = () => Navigator.pushNamed(context, '/signup'),
                            ),
                          ],
                        ),
                      ),
                      RichText(
                        text: TextSpan(
                          style: const TextStyle(fontSize: 13, color: AppColors.textGray),
                          children: [
                            const TextSpan(text: "I forgot the "),
                            TextSpan(
                              text: "password",
                              style: const TextStyle(color: AppColors.linkBlue),
                              recognizer: TapGestureRecognizer()
                                ..onTap = () => Navigator.pushNamed(context, '/forgot-password'),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 20),
                  
                  CustomButton(
                    text: "Log In",
                    isLoading: _isLoading,
                    onPressed: _handleLogin,
                  ),
                  
                  if (_bioAvailable) ...[
                    const SizedBox(height: 20),
                    const Center(
                      child: Text(
                        "or use biometrics",
                        style: TextStyle(
                          color: AppColors.textGray,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _buildBiometricButton(Icons.face_retouching_natural, _handleBiometricLogin),
                        const SizedBox(width: 24),
                        _buildBiometricButton(Icons.fingerprint, _handleBiometricLogin),
                      ],
                    ),
                  ],
                  
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBiometricButton(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 54,
        height: 54,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white24, width: 1),
        ),
        child: Center(
          child: Icon(icon, size: 28, color: Colors.white),
        ),
      ),
    );
  }
}
