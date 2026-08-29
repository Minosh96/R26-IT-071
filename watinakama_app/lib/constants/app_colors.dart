import 'package:flutter/material.dart';

class AppColors {
  // Header wave
  static const Color lightBlueTop = Color(0xFFBFD8F5);
  static const Color textDark = Color(0xFF12182A); // dark text on the light header

  // Backgrounds (light theme)
  static const Color darkNavyBg = Color(0xFFF7F7F5); // app background (off-white)
  static const Color darkNavyCard = Color(0xFFEFEFEC); // input fields
  static const Color darkNavySurface = Color(0xFFFFFFFF); // cards, sheets, dialogs
  static const Color darkNavySurfaceRaised = Color(0xFFE9E9E5); // hover/selected surface

  // Brand
  static const Color primaryBlue = Color(0xFF2F6FED);
  static const Color primaryBlueMuted = Color(0x332F6FED); // 20% primary, for chips/badges
  static const Color linkBlue = Color(0xFF1D5FE0);

  // Text (light theme: dark text on light surfaces)
  static const Color textWhite = Color(0xFF14171F); // primary text (kept name for call-site compatibility)
  static const Color textGray = Color(0xFF5B6472);
  static const Color textFaint = Color(0xFF9AA1AF);
  static const Color textFieldBorder = Color(0xFFD8DCE3);

  // Status
  static const Color statusGreen = Color(0xFF1A9850);
  static const Color statusAmber = Color(0xFFB77400);
  static const Color statusRed = Color(0xFFD92D20);

  // Status surfaces (subtle tinted backgrounds for banners/toasts)
  static const Color statusGreenBg = Color(0xFFE7F6EC);
  static const Color statusAmberBg = Color(0xFFFBF0DC);
  static const Color statusRedBg = Color(0xFFFBEAE9);

  static Color statusColorFor(double score) {
    if (score >= 80) return statusGreen;
    if (score >= 50) return statusAmber;
    return statusRed;
  }
}
