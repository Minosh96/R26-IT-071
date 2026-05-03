import 'package:flutter/material.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';

class VehicleImagesScreen extends StatefulWidget {
  const VehicleImagesScreen({super.key});

  @override
  State<VehicleImagesScreen> createState() => _VehicleImagesScreenState();
}

class _VehicleImagesScreenState extends State<VehicleImagesScreen> {
  final AuthService _auth = AuthService();
  Map<String, dynamic> _vehicleData = {};
  String _userName = '';
  String? _profilePicPath;

  @override
  void initState() {
    super.initState();
    _loadUserData();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;
    if (args is Map<String, dynamic>) {
      _vehicleData = args;
    }
  }

  Future<void> _loadUserData() async {
    final name = await _auth.getUserName();
    final pic = await _auth.getProfilePicPath();
    if (mounted) {
      setState(() {
        _userName = name;
        _profilePicPath = pic;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      appBar: InspectionAppBar(
        onBack: () => Navigator.pop(context),
        userName: _userName,
        userPhotoUrl: _profilePicPath,
      ),
      body: Column(
        children: [
          Container(
            height: 75,
            width: double.infinity,
            color: AppColors.lightBlueTop,
            alignment: Alignment.center,
            child: const ProgressStepper(currentStep: 2),
          ),
          
          Expanded(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.image_outlined, size: 80, color: AppColors.textGray),
                  const SizedBox(height: 20),
                  const Text(
                    "Vehicle Images",
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    "Step 3: Capture vehicle photos",
                    style: TextStyle(color: AppColors.textGray),
                  ),
                  const SizedBox(height: 40),
                  ElevatedButton(
                    onPressed: () {
                      // TODO: Implement image selection
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primaryBlue,
                      padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
                    ),
                    child: const Text("Capture Images"),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
