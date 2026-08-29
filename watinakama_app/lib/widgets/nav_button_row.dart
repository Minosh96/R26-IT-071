import 'package:flutter/material.dart';
import 'custom_button.dart';

/// The Back / Next (or Back / Analyze, Back / Confirm...) footer pair used
/// on every step of the inspection flow.
class NavButtonRow extends StatelessWidget {
  final VoidCallback onBack;
  final VoidCallback? onNext;
  final String backLabel;
  final String nextLabel;
  final bool isNextLoading;

  const NavButtonRow({
    super.key,
    required this.onBack,
    required this.onNext,
    this.backLabel = 'Back',
    this.nextLabel = 'Next',
    this.isNextLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        CustomButton(
          text: backLabel,
          onPressed: onBack,
          variant: AppButtonVariant.outline,
          fullWidth: false,
        ),
        CustomButton(
          text: nextLabel,
          onPressed: onNext,
          isLoading: isNextLoading,
          fullWidth: false,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
        ),
      ],
    );
  }
}
