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
import '../widgets/custom_text_field.dart';
import '../widgets/custom_button.dart';
import '../widgets/app_card.dart';
import '../services/biometric_service.dart';
import '../widgets/custom_toast.dart';
import '../utils/error_messages.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final BiometricService _bio = BiometricService();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();

  bool _isLoading = false;
  bool _showPasswordFields = false;
  bool _isBioAvailable = false;
  bool _isBioEnabled = false;
  File? _profileImage;
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
      _emailController.text = user.email ?? '';
      _userName = user.displayName ?? 'User';

      final prefs = await SharedPreferences.getInstance();
      _phoneController.text = prefs.getString('user_phone') ?? '';

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
      await user?.updateDisplayName(name);

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('user_phone', _phoneController.text.trim());
      await prefs.setString('user_name', name);

      if (_profileImage != null) {
        final Directory appDir = await getApplicationDocumentsDirectory();
        final String fileName = path.basename(_profileImage!.path);
        final File permanentImage = await _profileImage!.copy('${appDir.path}/$fileName');
        await prefs.setString('user_profile_pic', permanentImage.path);
      }

      if (_showPasswordFields && _passwordController.text.isNotEmpty) {
        await user?.updatePassword(_passwordController.text);

        const secureStorage = FlutterSecureStorage(
          aOptions: AndroidOptions(encryptedSharedPreferences: true),
        );
        await secureStorage.write(key: 'password', value: _passwordController.text);
      }

      if (mounted) {
        ToastService.show(context, "Profile updated successfully");
        Navigator.pop(context);
      }
    } on FirebaseAuthException catch (e) {
      _showError(firebaseAuthErrorMessage(e));
    } catch (e) {
      _showError(friendlyErrorMessage(e));
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
    _emailController.dispose();
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
        centerTitle: true,
        automaticallyImplyLeading: false,
        title: const Text(
          "Manage Profile",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: AppColors.textWhite),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            WaveHeader(
              height: 245,
              child: Center(
                child: Column(
                  children: [
                    const SizedBox(height: 30),
                    Stack(
                      children: [
                        CircleAvatar(
                          radius: 52,
                          backgroundColor: Colors.white,
                          backgroundImage: _profileImage != null ? FileImage(_profileImage!) : null,
                          child: _profileImage == null
                              ? const Icon(Icons.person, size: 52, color: Colors.grey)
                              : null,
                        ),
                        Positioned(
                          bottom: 0,
                          right: 0,
                          child: GestureDetector(
                            onTap: _pickImage,
                            child: Container(
                              width: 32,
                              height: 32,
                              decoration: BoxDecoration(
                                color: AppColors.primaryBlue,
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white, width: 2),
                              ),
                              child: const Icon(Icons.camera_alt, color: Colors.white, size: 16),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Hi, $_userName",
                      style: const TextStyle(color: AppColors.textDark, fontSize: 14, fontWeight: FontWeight.w500),
                    ),
                  ],
                ),
              ),
            ),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Your Account Details",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: AppColors.textWhite),
                  ),
                  const SizedBox(height: 20),

                  CustomTextField(
                    label: "Full Name",
                    controller: _nameController,
                    hint: "Enter your name",
                  ),
                  const SizedBox(height: 14),

                  CustomTextField(
                    label: "E-mail",
                    controller: _emailController,
                    enabled: false,
                    suffixIcon: Icons.lock_outline,
                  ),
                  const SizedBox(height: 14),

                  CustomTextField(
                    label: "Mobile Number",
                    controller: _phoneController,
                    keyboardType: TextInputType.phone,
                    prefixText: "+94",
                    hint: "7X XXX XXXX",
                  ),
                  const SizedBox(height: 20),

                  Row(
                    children: [
                      const Text("Password", style: TextStyle(color: AppColors.textGray, fontSize: 12)),
                      const SizedBox(width: 12),
                      if (!_showPasswordFields)
                        GestureDetector(
                          onTap: () => setState(() => _showPasswordFields = true),
                          child: const Text(
                            "Change password",
                            style: TextStyle(fontSize: 12, color: AppColors.linkBlue, decoration: TextDecoration.underline),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 6),

                  if (_showPasswordFields) ...[
                    CustomTextField(
                      label: "",
                      controller: _passwordController,
                      isPassword: true,
                      hint: "New password",
                    ),
                    const SizedBox(height: 12),
                    CustomTextField(
                      label: "",
                      controller: _confirmPasswordController,
                      isPassword: true,
                      hint: "Confirm new password",
                    ),
                  ] else
                    CustomTextField(
                      label: "",
                      controller: TextEditingController(text: '••••••••'),
                      enabled: false,
                    ),

                  const SizedBox(height: 28),
                  CustomButton(
                    text: "Save Changes",
                    onPressed: _handleSave,
                    isLoading: _isLoading,
                  ),
                  const SizedBox(height: 20),

                  if (_isBioAvailable) ...[
                    const Text("Security", style: TextStyle(color: AppColors.textGray, fontSize: 12)),
                    const SizedBox(height: 6),
                    AppCard(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.fingerprint, color: AppColors.textWhite, size: 20),
                              SizedBox(width: 12),
                              Text("Biometric Login", style: TextStyle(color: AppColors.textWhite, fontSize: 14)),
                            ],
                          ),
                          Switch(
                            value: _isBioEnabled,
                            activeThumbColor: AppColors.primaryBlue,
                            onChanged: (value) async {
                              await _bio.setEnabled(value);
                              if (!mounted) return;
                              setState(() => _isBioEnabled = value);
                              ToastService.show(
                                context,
                                value ? "Biometric login enabled" : "Biometric login disabled",
                              );
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
                        style: TextStyle(fontSize: 13, color: AppColors.textGray, decoration: TextDecoration.underline),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
