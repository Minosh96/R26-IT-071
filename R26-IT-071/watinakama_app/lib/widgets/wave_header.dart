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
        // Bottom layer: Light blue background
        Container(
          height: height,
          width: double.infinity,
          color: AppColors.lightBlueTop,
        ),
        
        // Middle layer: Custom wave transition to dark navy (only at the bottom)
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: CustomPaint(
            size: const Size(double.infinity, 80), // Keep the wave height controlled
            painter: HeaderWavePainter(color: AppColors.darkNavyBg),
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

class HeaderWavePainter extends CustomPainter {
  final Color color;
  HeaderWavePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final path = Path();
    // Start at bottom left
    path.moveTo(0, size.height);
    // Curve start
    path.lineTo(0, size.height * 0.4);
    
    // Refined smooth S-curve transition (more subtle)
    path.cubicTo(
      size.width * 0.3, size.height * 0.2, // Start slightly lower
      size.width * 0.6, size.height * 0.8, // End slightly higher
      size.width, size.height * 0.3,
    );
    
    path.lineTo(size.width, size.height);
    path.close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
