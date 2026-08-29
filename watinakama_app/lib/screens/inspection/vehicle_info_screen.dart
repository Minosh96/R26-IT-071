import 'package:flutter/material.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../services/auth_service.dart';
import '../../widgets/custom_toast.dart';

class VehicleInfoScreen extends StatefulWidget {
  const VehicleInfoScreen({super.key});

  @override
  State<VehicleInfoScreen> createState() => _VehicleInfoScreenState();
}

class _VehicleInfoScreenState extends State<VehicleInfoScreen> {
  final AuthService _auth = AuthService();
  final _makeController = TextEditingController();
  final _modelController = TextEditingController();
  final _ownersController = TextEditingController();
  final _mileageController = TextEditingController();
  final _listedPriceController = TextEditingController();
  
  int _selectedMafYear = 2015;
  int _selectedRegYear = 2015;
  String _selectedBNR = 'Brand New';
  bool _powerShutters = false;
  bool _powerMirrors = false;
  String _userName = '';
  String? _profilePicPath;

  @override
  void initState() {
    super.initState();
    _loadUserName();
  }

  Future<void> _loadUserName() async {
    final name = await _auth.getUserName();
    final pic = await _auth.getProfilePicPath();
    if (mounted) {
      setState(() {
        _userName = name;
        _profilePicPath = pic;
      });
    }
  }

  void _handleNext() {
    if (_makeController.text.isEmpty || 
        _modelController.text.isEmpty || 
        _ownersController.text.isEmpty || 
        _mileageController.text.isEmpty || 
        _listedPriceController.text.isEmpty) {
      ToastService.show(context, "Please fill in all required fields", isError: true);
      return;
    }

    Navigator.pushNamed(context, '/inspection/audio', arguments: {
      'make': _makeController.text,
      'model': _modelController.text,
      'maf_year': _selectedMafYear,
      'reg_year': _selectedRegYear,
      'previous_owners': int.tryParse(_ownersController.text) ?? 1,
      'mileage_km': int.tryParse(_mileageController.text) ?? 0,
      'listed_price_million': double.tryParse(_listedPriceController.text) ?? 0.0,
      'is_reconditioned': _selectedBNR == 'Reconditioned' ? 1 : 0,
      'power_shutters': _powerShutters ? 1 : 0,
      'power_mirrors': _powerMirrors ? 1 : 0,
    });
  }

  @override
  void dispose() {
    _makeController.dispose();
    _modelController.dispose();
    _ownersController.dispose();
    _mileageController.dispose();
    _listedPriceController.dispose();
    super.dispose();
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
          // Step Progress Bar Section
          Container(
            height: 75,
            width: double.infinity,
            color: AppColors.lightBlueTop,
            alignment: Alignment.center,
            child: const ProgressStepper(currentStep: 0),
          ),
          
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Enter your vehicle details",
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 20),
                  
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1A2035),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildLabel("Make"),
                        _buildTextField(_makeController, "Suzuki"),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Model"),
                        _buildTextField(_modelController, "Alto"),
                        const SizedBox(height: 14),
                        
                        Row(
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text("Year of Manufacture", style: TextStyle(color: AppColors.textGray, fontSize: 11)),
                                  const SizedBox(height: 6),
                                  _buildDropdown(
                                    value: _selectedMafYear,
                                    items: List.generate(6, (i) => 2012 + i),
                                    onChanged: (val) => setState(() => _selectedMafYear = val!),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text("Year of Register", style: TextStyle(color: AppColors.textGray, fontSize: 11)),
                                  const SizedBox(height: 6),
                                  _buildDropdown(
                                    value: _selectedRegYear,
                                    items: List.generate(14, (i) => 2012 + i),
                                    onChanged: (val) => setState(() => _selectedRegYear = val!),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Total Previous Owners"),
                        _buildTextField(_ownersController, "e.g. 2", keyboardType: TextInputType.number),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Mileage (km)"),
                        _buildTextField(_mileageController, "e.g. 85000", keyboardType: TextInputType.number),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Listed Price (Million LKR)"),
                        _buildTextField(_listedPriceController, "e.g. 3.89", keyboardType: const TextInputType.numberWithOptions(decimal: true)),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Vehicle Import Condition"),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            _buildConditionOption('Brand New'),
                            const SizedBox(width: 12),
                            _buildConditionOption('Reconditioned'),
                          ],
                        ),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Additional Features"),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            Expanded(
                              child: CheckboxListTile(
                                value: _powerShutters,
                                onChanged: (val) => setState(() => _powerShutters = val!),
                                title: const Text("Power Shutters", style: TextStyle(color: Colors.white, fontSize: 13)),
                                activeColor: AppColors.primaryBlue,
                                contentPadding: EdgeInsets.zero,
                                controlAffinity: ListTileControlAffinity.leading,
                              ),
                            ),
                            Expanded(
                              child: CheckboxListTile(
                                value: _powerMirrors,
                                onChanged: (val) => setState(() => _powerMirrors = val!),
                                title: const Text("Power Mirrors", style: TextStyle(color: Colors.white, fontSize: 13)),
                                activeColor: AppColors.primaryBlue,
                                contentPadding: EdgeInsets.zero,
                                controlAffinity: ListTileControlAffinity.leading,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 24),
                  
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      ElevatedButton(
                        onPressed: _handleNext,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF2E7D32),
                          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                        ),
                        child: const Text(
                          "Next",
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLabel(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Text(text, style: const TextStyle(color: AppColors.textGray, fontSize: 13)),
    );
  }

  Widget _buildTextField(TextEditingController controller, String hint, {TextInputType keyboardType = TextInputType.text}) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      style: const TextStyle(color: Colors.white),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.white24, fontSize: 14),
        filled: true,
        fillColor: AppColors.darkNavyCard,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
    );
  }

  Widget _buildDropdown({required int value, required List<int> items, required Function(int?) onChanged}) {
    return DropdownButtonFormField<int>(
      initialValue: value,
      items: items.map((year) => DropdownMenuItem(
        value: year,
        child: Text(year.toString(), style: const TextStyle(color: Colors.white, fontSize: 14)),
      )).toList(),
      onChanged: onChanged,
      dropdownColor: AppColors.darkNavyCard,
      decoration: InputDecoration(
        filled: true,
        fillColor: AppColors.darkNavyCard,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      ),
    );
  }

  Widget _buildConditionOption(String option) {
    bool isSelected = _selectedBNR == option;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _selectedBNR = option),
        child: Container(
          height: 44,
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF1E3A5F) : AppColors.darkNavyCard,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: isSelected ? AppColors.primaryBlue : AppColors.textFieldBorder,
              width: isSelected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              option,
              style: TextStyle(
                color: isSelected ? Colors.white : AppColors.textGray,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class StepWavePainter extends CustomPainter {
  final Color color;
  StepWavePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = color..style = PaintingStyle.fill;
    final path = Path();
    path.moveTo(0, size.height);
    path.lineTo(0, size.height * 0.4);
    path.cubicTo(
      size.width * 0.3, size.height * 0.1,
      size.width * 0.6, size.height * 0.9,
      size.width, size.height * 0.3,
    );
    path.lineTo(size.width, size.height);
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
