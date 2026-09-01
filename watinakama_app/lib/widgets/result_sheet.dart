import 'package:flutter/material.dart';
import '../constants/app_colors.dart';
import 'custom_button.dart';

/// Shared shell for the "analysis complete" bottom sheets shown after VIN,
/// body, and engine scans. Each screen supplies its own summary [content];
/// the shell owns the sheet chrome and the primary continue action so that
/// styling doesn't get re-implemented per screen.
Future<void> showResultSheet(
  BuildContext context, {
  required Widget content,
  required String ctaLabel,
  required VoidCallback onCta,
}) {
  return showModalBottomSheet(
    context: context,
    backgroundColor: AppColors.darkNavySurface,
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
    ),
    builder: (sheetContext) => ConstrainedBox(
      // Cap the sheet height so a long summary can't push the CTA off-screen;
      // the content scrolls while the continue button stays pinned and visible.
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(sheetContext).size.height * 0.85,
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: EdgeInsets.fromLTRB(24, 28, 24, MediaQuery.of(sheetContext).viewInsets.bottom + 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Flexible(
                child: SingleChildScrollView(child: content),
              ),
              const SizedBox(height: 28),
              CustomButton(text: ctaLabel, onPressed: onCta),
            ],
          ),
        ),
      ),
    ),
  );
}
