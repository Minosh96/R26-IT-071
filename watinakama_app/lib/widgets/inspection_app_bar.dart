import 'dart:io';
import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class InspectionAppBar extends StatelessWidget implements PreferredSizeWidget {
  final VoidCallback onBack;
  final String userName;
  final String? userPhotoUrl;

  const InspectionAppBar({
    super.key,
    required this.onBack,
    required this.userName,
    this.userPhotoUrl,
  });

  @override
  Size get preferredSize => const Size.fromHeight(70);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: AppColors.lightBlueTop,
      elevation: 0,
      automaticallyImplyLeading: false,
      leading: GestureDetector(
        onTap: onBack,
        child: Container(
          margin: const EdgeInsets.all(12),
          decoration: const BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
          ),
          child: const Center(
            child: Icon(
              Icons.arrow_back,
              color: AppColors.darkNavyBg,
              size: 18,
            ),
          ),
        ),
      ),
      actions: [
        Padding(
          padding: const EdgeInsets.only(right: 16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              CircleAvatar(
                radius: 18,
                backgroundColor: Colors.white24,
                backgroundImage: (userPhotoUrl != null && File(userPhotoUrl!).existsSync())
                    ? FileImage(File(userPhotoUrl!))
                    : null,
                child: (userPhotoUrl == null || !File(userPhotoUrl!).existsSync())
                    ? const Icon(Icons.person, color: AppColors.textDark, size: 20)
                    : null,
              ),
              const SizedBox(height: 2),
              Text(
                "Hi, $userName",
                style: const TextStyle(
                  fontSize: 11,
                  color: AppColors.textDark,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
