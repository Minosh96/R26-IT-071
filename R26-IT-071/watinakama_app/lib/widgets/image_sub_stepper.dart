import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class ImageSubStepper extends StatelessWidget {
  final int currentAngle; // 0=Front, 1=Rear, 2=Left, 3=Right, 4=Up
  final List<bool> capturedAngles; // which angles have photos
  final Function(int) onAngleSelected;

  const ImageSubStepper({
    super.key,
    required this.currentAngle,
    required this.capturedAngles,
    required this.onAngleSelected,
  });

  static const List<String> steps = ['Front', 'Rear', 'Left', 'Right', 'Up'];

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2035),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: List.generate(steps.length, (i) {
          final isCaptured = i < capturedAngles.length && capturedAngles[i];
          final isCurrent = i == currentAngle;

          Color boxColor;
          Border boxBorder;
          Widget? boxChild;
          Color labelColor;

          if (isCaptured) {
            boxColor = const Color(0xFF1A3A2A);
            boxBorder = Border.all(
              color: isCurrent ? AppColors.primaryBlue : AppColors.statusGreen, 
              width: 2,
            );
            boxChild = const Icon(Icons.check, color: AppColors.statusGreen, size: 20);
            labelColor = isCurrent ? AppColors.primaryBlue : AppColors.statusGreen;
          } else if (isCurrent) {
            boxColor = const Color(0xFF1A2A3A);
            boxBorder = Border.all(color: AppColors.primaryBlue, width: 2);
            boxChild = const Text(
              "...",
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            );
            labelColor = AppColors.primaryBlue;
          } else {
            boxColor = const Color(0xFF1A2035);
            boxBorder = Border.all(color: AppColors.textFieldBorder, width: 1);
            boxChild = null;
            labelColor = AppColors.textGray;
          }

          return GestureDetector(
            onTap: () => onAngleSelected(i),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: boxColor,
                    border: boxBorder,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: boxChild != null ? Center(child: boxChild) : null,
                ),
                const SizedBox(height: 4),
                Text(
                  steps[i],
                  style: TextStyle(
                    color: labelColor,
                    fontSize: 10,
                    fontWeight: isCurrent ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ],
            ),
          );
        }),
      ),
    );
  }
}
