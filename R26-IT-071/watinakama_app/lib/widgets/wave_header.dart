import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class WaveHeader extends StatelessWidget {
  final double height;

  const WaveHeader({super.key, this.height = 280});

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
        
        // Middle layer: Custom wave transition to dark navy
        CustomPaint(
          size: Size(double.infinity, height),
          painter: WavePainter(color: AppColors.darkNavyBg),
        ),
        
        // Top layer: Centered content
        SizedBox(
          height: height,
          width: double.infinity,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              const SizedBox(height: 35), // Move up further
              
              // Car Logo Image
              Image.asset(
                'assets/images/car_logo.png',
                height: 140,
                width: 140,
                fit: BoxFit.contain,
              ),
              
              const SizedBox(height: 4),
              
              // App Name
              const Text(
                "වටිනාකම.LK",
                style: TextStyle(
                  fontSize: 22, // Slightly smaller text
                  fontWeight: FontWeight.bold,
                  color: AppColors.textDark,
                  fontFamily: 'Noto Sans Sinhala',
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class WavePainter extends CustomPainter {
  final Color color;
  WavePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final path = Path();
    // Start from bottom left
    path.moveTo(0, size.height);
    // Move up slightly to start the curve
    path.lineTo(0, size.height - 20);
    // Lowered peak further to 45px
    path.quadraticBezierTo(
      size.width / 2, 
      size.height - 45, // Lowered from 60 to 45
      size.width, 
      size.height - 20
    );
    // Close the shape at the bottom
    path.lineTo(size.width, size.height);
    path.close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
