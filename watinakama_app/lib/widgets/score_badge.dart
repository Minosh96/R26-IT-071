import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

/// A circular 0-100 score readout, used for the body-condition and
/// mechanical-health scores. Color follows the shared green/amber/red
/// threshold in [AppColors.statusColorFor].
class ScoreBadge extends StatelessWidget {
  final double score;
  final String label;
  final double size;

  const ScoreBadge({
    super.key,
    required this.score,
    required this.label,
    this.size = 100,
  });

  @override
  Widget build(BuildContext context) {
    final color = AppColors.statusColorFor(score);
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color.withValues(alpha: 0.12),
        border: Border.all(color: color, width: 3),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            score.toInt().toString(),
            style: TextStyle(fontSize: size * 0.28, fontWeight: FontWeight.bold, color: color),
          ),
          Text(
            label,
            style: const TextStyle(color: AppColors.textGray, fontSize: 12),
          ),
        ],
      ),
    );
  }
}
