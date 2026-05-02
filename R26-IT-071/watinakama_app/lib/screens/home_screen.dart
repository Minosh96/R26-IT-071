import 'package:flutter/material.dart';
import 'dart:io';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';
import '../widgets/custom_toast.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AuthService _auth = AuthService();
  String _userName = "User";
  String? _profilePicPath;
  DateTime? _lastPressedAt;

  final List<Map<String, String>> _inspections = [
    {"model": "Suzuki Alto", "date": "2026-04-15", "reg": "WP CAA 4512"},
    {"model": "Suzuki Alto", "date": "2026-04-15", "reg": "WP ABC 4812"},
    {"model": "Suzuki Alto", "date": "2026-04-15", "reg": "WP ABC 4812"},
  ];

  @override
  void initState() {
    super.initState();
    _loadUserName();
  }

  Future<void> _loadUserName() async {
    final name = await _auth.getUserName();
    
    final prefs = await SharedPreferences.getInstance();
    final picPath = prefs.getString('user_profile_pic');
    
    if (mounted) {
      setState(() {
        _userName = name;
        _profilePicPath = picPath;
      });
    }
  }

  Future<void> _handleLogout() async {
    final bool? confirm = await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.darkNavySurface,
        title: const Text("Logout", style: TextStyle(color: Colors.white)),
        content: const Text("Are you sure you want to logout?", style: TextStyle(color: AppColors.textGray)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel", style: TextStyle(color: Colors.white70)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text("Logout", style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await _auth.logout();
      if (mounted) {
        Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        
        final now = DateTime.now();
        final backButtonHasNotBeenPressedOrSnackBarHasBeenClosed =
            _lastPressedAt == null || now.difference(_lastPressedAt!) > const Duration(seconds: 2);

        if (backButtonHasNotBeenPressedOrSnackBarHasBeenClosed) {
          _lastPressedAt = now;
          ToastService.show(context, "Press back again to logout");
        } else {
          _handleLogout();
        }
      },
      child: Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Custom Header with Wave and Profile info
            WaveHeader(
              height: 120,
              child: Column(
                children: [
                  // Profile & Logout Actions Row
                  Padding(
                    padding: const EdgeInsets.only(top: 35, right: 16, left: 16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.logout, color: AppColors.textDark, size: 20),
                          onPressed: _handleLogout,
                        ),
                        GestureDetector(
                          onTap: () async {
                            await Navigator.pushNamed(context, '/profile');
                            _loadUserName(); // Refresh name and picture when returning
                          },
                          child: Row(
                            children: [
                              CircleAvatar(
                                radius: 18,
                                backgroundColor: Colors.white24,
                                backgroundImage: (_profilePicPath != null && File(_profilePicPath!).existsSync())
                                    ? FileImage(File(_profilePicPath!))
                                    : null,
                                child: (_profilePicPath == null || !File(_profilePicPath!).existsSync())
                                    ? const Icon(Icons.person, color: AppColors.textDark, size: 20)
                                    : null,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                "Hi, $_userName",
                                style: const TextStyle(
                                  color: AppColors.textDark,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            
            // Search Bar Section
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: "Enter Vehicle Reg No",
                        hintStyle: const TextStyle(color: AppColors.textGray, fontSize: 14),
                        filled: true,
                        fillColor: const Color(0xFF1A2035),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 90,
                    height: 48,
                    child: ElevatedButton(
                      onPressed: () {},
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primaryBlue,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        elevation: 0,
                      ),
                      child: const Text(
                        "Search",
                        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            // Inspection Heading
            const Padding(
              padding: EdgeInsets.only(top: 10, bottom: 20),
              child: Text(
                "Let's inspect your vehicle",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
            
            // Start Button
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: CustomButton(
                text: "Start",
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Inspection flow coming soon")),
                  );
                },
              ),
            ),
            
            const SizedBox(height: 40),
            
            // Recent Inspections List
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Recent Inspections",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  ListView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: _inspections.length,
                    itemBuilder: (context, index) {
                      final inspection = _inspections[index];
                      return Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1A2035),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  inspection["model"]!,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  inspection["date"]!,
                                  style: const TextStyle(
                                    color: AppColors.textGray,
                                    fontSize: 12,
                                  ),
                                ),
                              ],
                            ),
                            Text(
                              inspection["reg"]!,
                              style: const TextStyle(
                                color: AppColors.textGray,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    ),
    );
  }
}
