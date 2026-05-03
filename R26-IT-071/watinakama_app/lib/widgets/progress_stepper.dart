import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class ProgressStepper extends StatelessWidget {
  final int currentStep; // 0=Info, 1=Audio, 2=Images, 3=VIN, 4=Results

  const ProgressStepper({super.key, required this.currentStep});

  @override
  Widget build(BuildContext context) {
    final steps = ['Info', 'Audio', 'Images', 'VIN', 'Results'];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: List.generate(steps.length, (index) {
          return Expanded(
            child: Row(
              children: [
                // Step Indicator & Label
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildCircle(index),
                    const SizedBox(height: 4),
                    Text(
                      steps[index],
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: index <= currentStep ? FontWeight.bold : FontWeight.normal,
                        color: index <= currentStep ? AppColors.primaryBlue : AppColors.textGray,
                      ),
                    ),
                  ],
                ),
                
                // Connector Line (only if not the last step)
                if (index < steps.length - 1)
                  Expanded(
                    child: Container(
                      height: 2,
                      margin: const EdgeInsets.only(bottom: 14), // Align with circles
                      color: index < currentStep 
                          ? AppColors.primaryBlue 
                          : AppColors.textGray.withOpacity(0.3),
                    ),
                  ),
              ],
            ),
          );
        }),
      ),
    );
  }

  Widget _buildCircle(int index) {
    bool isCompleted = index < currentStep;
    bool isActive = index == currentStep;

    return Container(
      width: 24,
      height: 24,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: (isCompleted || isActive) ? AppColors.primaryBlue : Colors.transparent,
        border: Border.all(
          color: (isCompleted || isActive) ? AppColors.primaryBlue : AppColors.textGray,
          width: 1.5,
        ),
      ),
      child: Center(
        child: isCompleted
            ? const Icon(Icons.check, size: 14, color: Colors.white)
            : isActive
                ? Container(
                    width: 6,
                    height: 6,
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                    ),
                  )
                : Text(
                    (index + 1).toString(),
                    style: const TextStyle(
                      fontSize: 10,
                      color: AppColors.textGray,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
      ),
    );
  }
}
