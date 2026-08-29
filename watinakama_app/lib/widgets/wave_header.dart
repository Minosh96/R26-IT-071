import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class WaveHeader extends StatelessWidget {
  final double height;
  final Widget? child;

  const WaveHeader({super.key, this.height = 280, this.child});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Bottom layer: Light blue background with a straight bottom edge
        Container(
          height: height,
          width: double.infinity,
          decoration: const BoxDecoration(
            color: AppColors.lightBlueTop,
            border: Border(
              bottom: BorderSide(color: AppColors.darkNavyBg, width: 2),
            ),
          ),
        ),

        // Top layer: Content
        SizedBox(
          height: height,
          width: double.infinity,
          child: child ?? _buildDefaultContent(),
        ),
      ],
    );
  }

  Widget _buildDefaultContent() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.start,
      children: [
        const SizedBox(height: 30), // Slightly less top padding
        // Car Logo Image
        Image.asset(
          'assets/images/car_logo.png',
          height: 120, // Reduced from 140
          width: 120,
          fit: BoxFit.contain,
        ),
        const SizedBox(height: 2), // Tighter spacing
        // App Name
        const Text(
          "වටිනාකම.LK",
          style: TextStyle(
            fontSize: 20, // Slightly smaller text
            fontWeight: FontWeight.bold,
            color: AppColors.textDark,
            fontFamily: 'Noto Sans Sinhala',
          ),
        ),
      ],
    );
  }
}
