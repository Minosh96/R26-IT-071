import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/splash_screen.dart';
import 'screens/login_screen.dart';
import 'screens/signup_screen.dart';
import 'screens/home_screen.dart';
import 'screens/forgot_password_screen.dart';
import 'screens/profile_screen.dart';
import 'constants/app_colors.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const WatinakamaApp());
}

class WatinakamaApp extends StatelessWidget {
  const WatinakamaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'වටිනාකම.LK',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AppColors.darkNavyBg,
        textTheme: GoogleFonts.interTextTheme(
          ThemeData.dark().textTheme
        ).apply(
          fontFamily: 'Noto Sans Sinhala',
        ),
        colorScheme: ColorScheme.dark(
          primary: AppColors.primaryBlue,
          surface: AppColors.darkNavySurface,
          secondary: AppColors.linkBlue,
        ),
      ),
      initialRoute: '/splash',
      routes: {
        '/splash': (context) => const SplashScreen(),
        '/login':  (context) => const LoginScreen(),
        '/signup': (context) => const SignUpScreen(),
        '/forgot-password': (context) => const ForgotPasswordScreen(),
        '/home':   (context) => const HomeScreen(),
        '/profile': (context) => const ProfileScreen(),
      },
    );
  }
}

