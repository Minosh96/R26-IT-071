import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

enum AppButtonVariant { primary, outline }

class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final bool isLoading;
  final IconData? suffixIcon;
  final AppButtonVariant variant;
  final bool fullWidth;
  final EdgeInsetsGeometry padding;

  const CustomButton({
    super.key,
    required this.text,
    required this.onPressed,
    this.isLoading = false,
    this.suffixIcon,
    this.variant = AppButtonVariant.primary,
    this.fullWidth = true,
    this.padding = const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
  });

  bool get _isOutline => variant == AppButtonVariant.outline;

  @override
  Widget build(BuildContext context) {
    final button = ElevatedButton(
      onPressed: isLoading ? null : onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: _isOutline ? AppColors.darkNavyCard : AppColors.primaryBlue,
        foregroundColor: _isOutline ? AppColors.textWhite : Colors.white,
        disabledBackgroundColor: _isOutline
            ? AppColors.darkNavyCard
            : AppColors.primaryBlue.withValues(alpha: 0.6),
        padding: padding,
        side: _isOutline ? const BorderSide(color: AppColors.textFieldBorder) : BorderSide.none,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
        ),
        elevation: 0,
      ),
      child: isLoading
          ? SizedBox(
              height: 18,
              width: 18,
              child: CircularProgressIndicator(
                color: _isOutline ? AppColors.textWhite : Colors.white,
                strokeWidth: 2,
              ),
            )
          : Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  text,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.3,
                  ),
                ),
                if (suffixIcon != null) ...[
                  const SizedBox(width: 10),
                  Icon(suffixIcon, size: 18),
                ],
              ],
            ),
    );

    if (!fullWidth) return button;

    return SizedBox(
      width: double.infinity,
      height: 44,
      child: button,
    );
  }
}
