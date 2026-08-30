import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import '../services/auth_service.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  final AuthService _authService = AuthService();

  @override
  void initState() {
    super.initState();
    
    // Animation controller for 3 seconds
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    );

    _animation = Tween<double>(begin: 0, end: 1).animate(_controller)
      ..addListener(() {
        setState(() {});
      });

    _controller.forward().then((value) => _navigateToNext());
  }

  Future<void> _navigateToNext() async {
    final isLoggedIn = await _authService.isLoggedIn();
    if (mounted) {
      Navigator.pushReplacementNamed(
        context,
        isLoggedIn ? '/home' : '/login',
      );
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.lightBlueTop,
      body: Stack(
        children: [
          // Centered Content
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset(
                  'assets/images/car_logo.png',
                  height: 200,
                  width: 200,
                  fit: BoxFit.contain,
                ),
                const SizedBox(height: 24),
                const Text(
                  "වටිනාකම.LK",
                  style: TextStyle(
                    color: AppColors.textDark,
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'Noto Sans Sinhala',
                  ),
                ),
              ],
            ),
          ),
          
          // Progress Bar at the bottom
          Positioned(
            bottom: 60,
            left: 50,
            right: 50,
            child: Column(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: LinearProgressIndicator(
                    value: _animation.value,
                    backgroundColor: Colors.white24,
                    valueColor: const AlwaysStoppedAnimation<Color>(AppColors.primaryBlue),
                    minHeight: 6,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  "${(_animation.value * 100).toInt()}%",
                  style: const TextStyle(
                    color: AppColors.textDark,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
