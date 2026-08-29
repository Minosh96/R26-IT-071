import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

/// The standard rounded surface used for grouped content across the app
/// (form sections, result summaries, info rows). Centralizes the
/// card background/radius that used to be copy-pasted as
/// `Color(0xFF1A2035)` + `BorderRadius.circular(16)` in a dozen places.
class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color? borderColor;
  final double radius;

  const AppCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.borderColor,
    this.radius = 16,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: AppColors.darkNavySurface,
        borderRadius: BorderRadius.circular(radius),
        border: borderColor != null ? Border.all(color: borderColor!, width: 1.5) : null,
      ),
      child: child,
    );
  }
}
