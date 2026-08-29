import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants/app_colors.dart';
import '../widgets/wave_header.dart';
import '../widgets/custom_button.dart';
import '../services/auth_service.dart';
import '../services/biometric_service.dart';
import '../widgets/custom_toast.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final AuthService _auth = AuthService();
  final BiometricService _bio = BiometricService();
  final _nameController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  
  bool _isLoading = false;
  bool _showPasswordFields = false;
  bool _isBioAvailable = false;
  bool _isBioEnabled = false;
  File? _profileImage;
  String _userEmail = '';
  String _userName = '';

  @override
  void initState() {
    super.initState();
    _loadUserData();
    _checkBiometrics();
  }

  Future<void> _checkBiometrics() async {
    _isBioAvailable = await _bio.isAvailable();
    _isBioEnabled = await _bio.isEnabled();
    if (mounted) setState(() {});
  }

  Future<void> _loadUserData() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      _nameController.text = user.displayName ?? '';
      _userEmail = user.email ?? '';
      _userName = user.displayName ?? 'User';
      
      final prefs = await SharedPreferences.getInstance();
      _phoneController.text = prefs.getString('user_phone') ?? '';
      
      // Load profile image path
      final imagePath = prefs.getString('user_profile_pic');
      if (imagePath != null && File(imagePath).existsSync()) {
        _profileImage = File(imagePath);
      }
      
      setState(() {});
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
    if (picked != null) {
      setState(() => _profileImage = File(picked.path));
    }
  }

  Future<void> _handleSave() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      _showError("Name cannot be empty");
      return;
    }

    if (_showPasswordFields) {
      final pass = _passwordController.text;
      final confirm = _confirmPasswordController.text;
      
      if (pass.isEmpty) {
        _showError("Please enter a new password");
        return;
      }
      if (pass != confirm) {
        _showError("Passwords do not match");
        return;
      }
      if (pass.length < 8) {
        _showError("Password must be at least 8 characters");
        return;
      }
    }

    setState(() => _isLoading = true);

    try {
      final user = FirebaseAuth.instance.currentUser;
      
      // Update display name
      await user?.updateDisplayName(name);
      
      // Save phone to prefs
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('user_phone', _phoneController.text.trim());
      await prefs.setString('user_name', name); // Keep synced with AuthService logic

      // Handle Image Persistence (Local Storage)
      if (_profileImage != null) {
        final Directory appDir = await getApplicationDocumentsDirectory();
        final String fileName = path.basename(_profileImage!.path);
        final File permanentImage = await _profileImage!.copy('${appDir.path}/$fileName');
        await prefs.setString('user_profile_pic', permanentImage.path);
      }

      // Update password if requested
      if (_showPasswordFields && _passwordController.text.isNotEmpty) {
        await user?.updatePassword(_passwordController.text);
        
        // Sync with secure storage for biometric login
        const secureStorage = FlutterSecureStorage(
          aOptions: AndroidOptions(
            encryptedSharedPreferences: true,
          ),
        );
        await secureStorage.write(key: 'password', value: _passwordController.text);
      }

      if (mounted) {
        ToastService.show(context, "Profile updated successfully");
        Navigator.pop(context);
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == 'requires-recent-login') {
        _showError("Please logout and login again before changing password");
      } else {
        _showError(e.message ?? "An error occurred");
      }
    } catch (e) {
      _showError(e.toString());
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showError(String message) {
    ToastService.show(context, message, isError: true);
  }

  @override
  void dispose() {
    _nameController.dispose();
    _phoneController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkNavyBg,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        automaticallyImplyLeading: false,
        title: const Text(
          "Manage Profile",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: Colors.white),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Top Section with Wave
            WaveHeader(
              height: 240,
              child: Center(
                child: Column(
                  children: [
                    const SizedBox(height: 40),
                    Stack(
                      children: [
                        CircleAvatar(
                          radius: 55,
                          backgroundColor: Colors.white,
                          backgroundImage: _profileImage != null
                              ? FileImage(_profileImage!)
                              : null,
                          child: _profileImage == null
                              ? const Icon(Icons.person, size: 55, color: Colors.grey)
                              : null,
                        ),
                        Positioned(
                          bottom: 0,
                          right: 0,
                          child: GestureDetector(
                            onTap: _pickImage,
                            child: Container(
                              width: 34,
                              height: 34,
                              decoration: BoxDecoration(
                                color: AppColors.primaryBlue,
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white, width: 2),
                              ),
                              child: const Icon(Icons.camera_alt, color: Colors.white, size: 18),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Hi, $_userName",
                      style: const TextStyle(
                        color: AppColors.textDark,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Your Account Details",
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const SizedBox(height: 20),

                  // Full Name
                  _buildLabel("Full Name (Username)"),
                  _buildTextField(_nameController, hint: "Enter your name"),
                  const SizedBox(height: 16),

                  // Email (Disabled)
                  _buildLabel("E-mail (Disabled)"),
                  TextField(
                    controller: TextEditingController(text: _userEmail),
                    enabled: false,
                    style: const TextStyle(color: AppColors.textGray),
                    decoration: InputDecoration(
                      filled: true,
                      fillColor: const Color(0xFF1A1A2A),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      suffixIcon: const Icon(Icons.lock_outline, color: AppColors.textGray, size: 18),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: const BorderSide(color: AppColors.textFieldBorder),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Mobile Number
                  _buildLabel("Mobile Number"),
                  Row(
                    children: [
                      Container(
                        width: 65,
                        height: 52,
                        decoration: BoxDecoration(
                          color: AppColors.darkNavyCard,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: AppColors.textFieldBorder),
                        ),
                        child: const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text("🇱🇰", style: TextStyle(fontSize: 18)),
                            SizedBox(width: 4),
                            Icon(Icons.arrow_drop_down, color: AppColors.textGray, size: 16),
                          ],
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: _buildTextField(_phoneController, hint: "+94 77 XXXXXXX", keyboardType: TextInputType.phone),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),

                  // Password Section
                  Row(
                    children: [
                      _buildLabel("Password"),
                      const SizedBox(width: 12),
                      GestureDetector(
                        onTap: () => setState(() => _showPasswordFields = !_showPasswordFields),
                        child: const Text(
                          "New Password",
                          style: TextStyle(
                            fontSize: 12,
                            color: AppColors.linkBlue,
                            decoration: TextDecoration.underline,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  
                  if (_showPasswordFields) ...[
                    _buildTextField(_passwordController, hint: "New Password", isPassword: true),
                    const SizedBox(height: 14),
                    _buildLabel("Confirm New Password"),
                    _buildTextField(_confirmPasswordController, hint: "Confirm New Password", isPassword: true),
                  ] else ...[
                    TextField(
                      controller: TextEditingController(text: '••••••••'),
                      enabled: false,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: AppColors.darkNavyCard,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10),
                          borderSide: const BorderSide(color: AppColors.textFieldBorder),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    GestureDetector(
                      onTap: () => setState(() => _showPasswordFields = true),
                      child: const Text(
                        "Need to change password?",
                        style: TextStyle(
                          fontSize: 13,
                          color: AppColors.linkBlue,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ),
                  ],

                  const SizedBox(height: 28),
                  CustomButton(
                    text: "Save Changes",
                    onPressed: _handleSave,
                    isLoading: _isLoading,
                  ),
                  const SizedBox(height: 20),
                  
                  if (_isBioAvailable) ...[
                    _buildLabel("Security"),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: AppColors.darkNavyCard,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: AppColors.textFieldBorder),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.fingerprint, color: Colors.white, size: 20),
                              SizedBox(width: 12),
                              Text("Biometric Login", style: TextStyle(color: Colors.white, fontSize: 14)),
                            ],
                          ),
                          Switch(
                            value: _isBioEnabled,
                            activeThumbColor: AppColors.primaryBlue,
                            onChanged: (value) async {
                              await _bio.setEnabled(value);
                              setState(() => _isBioEnabled = value);
                              if (!value) {
                                // If disabling, we could clear secure storage, 
                                // but logout handles that too if bioEnabled is false.
                                // For now just changing the preference is enough.
                                ToastService.show(context, "Biometric login disabled");
                              } else {
                                ToastService.show(context, "Biometric login enabled");
                              }
                            },
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 14),
                  ],

                  Center(
                    child: GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: const Text(
                        "Cancel & Back to Home",
                        style: TextStyle(
                          fontSize: 13,
                          color: AppColors.textGray,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLabel(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Text(
        text,
        style: const TextStyle(fontSize: 13, color: AppColors.textGray),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller,
      {String? hint, bool isPassword = false, TextInputType? keyboardType}) {
    return TextField(
      controller: controller,
      obscureText: isPassword,
      keyboardType: keyboardType,
      style: const TextStyle(color: Colors.white),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.white24),
        filled: true,
        fillColor: AppColors.darkNavyCard,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.textFieldBorder),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.primaryBlue),
        ),
      ),
    );
  }
}

