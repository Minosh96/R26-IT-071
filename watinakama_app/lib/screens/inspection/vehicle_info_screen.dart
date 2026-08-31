import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../constants/app_colors.dart';
import '../../widgets/inspection_app_bar.dart';
import '../../widgets/progress_stepper.dart';
import '../../widgets/app_card.dart';
import '../../widgets/custom_button.dart';
import '../../services/auth_service.dart';
import '../../widgets/custom_toast.dart';

class VehicleInfoScreen extends StatefulWidget {
  const VehicleInfoScreen({super.key});

  @override
  State<VehicleInfoScreen> createState() => _VehicleInfoScreenState();
}

class _VehicleInfoScreenState extends State<VehicleInfoScreen> {
  final AuthService _auth = AuthService();
  final TextEditingController _ownersController = TextEditingController();
  final TextEditingController _mileageController = TextEditingController();
  final TextEditingController _listedPriceController = TextEditingController();
  
  String _selectedMake = '';
  String _selectedModel = '';
  
  final Map<String, List<String>> _vehicleDataMap = {
    'Toyota': ['Corolla', 'Camry', 'Prius', 'Yaris', 'Aqua', 'Hilux', 'Fortuner', 'Land Cruiser', 'Prado', 'Vitz', 'Raize'],
    'Suzuki': ['Alto', 'Wagon R', 'Swift', 'Baleno', 'Celerio', 'Ignis', 'Vitara', 'Grand Vitara', 'Brezza', 'Ertiga', 'XL6', 'Jimny', 'Spacia', 'Hustler', 'Carry'],
    'Honda': ['Civic', 'Accord', 'Fit (Jazz)', 'City', 'CR-V', 'HR-V', 'BR-V', 'Insight', 'Freed'],
    'Nissan': ['Sunny', 'Altima', 'Teana', 'X-Trail', 'Qashqai', 'Patrol', 'Leaf', 'Navara', 'Juke'],
    'Mitsubishi Motors': ['Lancer', 'Outlander', 'Pajero', 'Montero Sport', 'Eclipse Cross', 'Mirage'],
    'Mazda': ['Mazda2', 'Mazda3', 'Mazda6', 'CX-3', 'CX-5', 'CX-8', 'CX-9'],
    'Hyundai': ['i10', 'i20', 'i30', 'Accent', 'Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Creta'],
    'Kia': ['Picanto', 'Rio', 'Cerato', 'Seltos', 'Sportage', 'Sorento', 'Carnival'],
    'Ford': ['Fiesta', 'Focus', 'Mustang', 'Ranger', 'Everest', 'Explorer', 'F-150'],
    'Chevrolet': ['Spark', 'Aveo', 'Cruze', 'Malibu', 'Tahoe', 'Silverado'],
    'BMW': ['1 Series', '3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7'],
    'Mercedes-Benz': ['A-Class', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE', 'GLS', 'G-Class'],
    'Audi': ['A3', 'A4', 'A6', 'A8', 'Q2', 'Q3', 'Q5', 'Q7'],
    'Volkswagen': ['Polo', 'Golf', 'Passat', 'Jetta', 'Tiguan', 'Touareg'],
    'Tata Motors': ['Nano', 'Tiago', 'Tigor', 'Nexon', 'Harrier', 'Safari'],
    'Mahindra': ['Bolero', 'Scorpio', 'Thar', 'XUV300', 'XUV500', 'XUV700'],
    'Tesla': ['Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck'],
    'Subaru': ['Impreza', 'Legacy', 'Forester', 'Outback', 'XV (Crosstrek)'],
    'Isuzu': ['D-Max', 'MU-X', 'Elf (truck)'],
    'Peugeot': ['208', '308', '508', '2008', '3008', '5008'],
  };

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
    // TODO: re-enable required-field validation once testing is done.

    Navigator.pushNamed(context, '/inspection/audio', arguments: {
      'make': _selectedMake,
      'model': _selectedModel,

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
    _ownersController.dispose();
    _mileageController.dispose();
    _listedPriceController.dispose();
    super.dispose();
  }

  void _showSearchableDropdown({
    required String title,
    required List<String> items,
    required Function(String) onSelected,
  }) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.darkNavySurface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return _SearchableListModal(
          title: title,
          items: items,
          onSelected: onSelected,
        );
      },
    );
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
                      color: AppColors.textWhite,
                    ),
                  ),
                  const SizedBox(height: 20),
                  
                  AppCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildLabel("Make"),
                        _buildSelectionField(
                          value: _selectedMake,
                          hint: "Select Make",
                          onTap: () {
                            _showSearchableDropdown(
                              title: "Select Vehicle Make",
                              items: _vehicleDataMap.keys.toList()..sort(),
                              onSelected: (val) {
                                setState(() {
                                  _selectedMake = val;
                                  _selectedModel = ''; // Reset model when make changes
                                });
                              },
                            );
                          },
                        ),
                        const SizedBox(height: 14),
                        
                        _buildLabel("Model"),
                        _buildSelectionField(
                          value: _selectedModel,
                          hint: _selectedMake.isEmpty ? "Select Make first" : "Select Model",
                          onTap: _selectedMake.isEmpty 
                            ? null 
                            : () {
                                _showSearchableDropdown(
                                  title: "Select $_selectedMake Model",
                                  items: _vehicleDataMap[_selectedMake]!..sort(),
                                  onSelected: (val) {
                                    setState(() => _selectedModel = val);
                                  },
                                );
                              },
                        ),
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
                        _buildTextField(
                          _ownersController,
                          "e.g. 2",
                          keyboardType: TextInputType.number,
                          inputFormatters: [
                            FilteringTextInputFormatter.digitsOnly,
                            LengthLimitingTextInputFormatter(2),
                          ],
                        ),
                        const SizedBox(height: 14),

                        _buildLabel("Mileage (km)"),
                        _buildTextField(
                          _mileageController,
                          "e.g. 85000",
                          keyboardType: TextInputType.number,
                          inputFormatters: [
                            FilteringTextInputFormatter.digitsOnly,
                            LengthLimitingTextInputFormatter(7),
                          ],
                        ),
                        const SizedBox(height: 14),

                        _buildLabel("Listed Price (Million LKR)"),
                        _buildTextField(
                          _listedPriceController,
                          "e.g. 3.89",
                          keyboardType: const TextInputType.numberWithOptions(decimal: true),
                          inputFormatters: [_decimalInputFormatter],
                        ),
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
                                title: const Text("Power Shutters", style: TextStyle(color: AppColors.textWhite, fontSize: 13)),
                                activeColor: AppColors.primaryBlue,
                                contentPadding: EdgeInsets.zero,
                                controlAffinity: ListTileControlAffinity.leading,
                              ),
                            ),
                            Expanded(
                              child: CheckboxListTile(
                                value: _powerMirrors,
                                onChanged: (val) => setState(() => _powerMirrors = val!),
                                title: const Text("Power Mirrors", style: TextStyle(color: AppColors.textWhite, fontSize: 13)),
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
                      CustomButton(
                        text: "Next",
                        onPressed: _handleNext,
                        fullWidth: false,
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
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

  Widget _buildSelectionField({required String value, required String hint, VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.darkNavyCard,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.textFieldBorder.withValues(alpha: 0.5)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              value.isEmpty ? hint : value,
              style: TextStyle(
                color: value.isEmpty ? AppColors.textFaint : AppColors.textWhite,
                fontSize: 14,
              ),
            ),
            const Icon(Icons.keyboard_arrow_down, color: AppColors.textGray, size: 20),
          ],
        ),
      ),
    );
  }

  static final _decimalInputFormatter = TextInputFormatter.withFunction((oldValue, newValue) {
    if (newValue.text.isEmpty) return newValue;
    return RegExp(r'^\d{0,7}(\.\d{0,2})?$').hasMatch(newValue.text) ? newValue : oldValue;
  });

  Widget _buildTextField(
    TextEditingController controller,
    String hint, {
    TextInputType keyboardType = TextInputType.text,
    List<TextInputFormatter>? inputFormatters,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      inputFormatters: inputFormatters,
      style: const TextStyle(color: AppColors.textWhite),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: AppColors.textFaint, fontSize: 14),
        filled: true,
        fillColor: AppColors.darkNavyCard,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
    );
  }

  Widget _buildDropdown({required int value, required List<int> items, required Function(int?) onChanged}) {
    return DropdownButtonFormField<int>(
      value: value,
      items: items.map((year) => DropdownMenuItem(
        value: year,
        child: Text(year.toString(), style: const TextStyle(color: AppColors.textWhite, fontSize: 14)),
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
            color: isSelected ? AppColors.primaryBlueMuted : AppColors.darkNavyCard,
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
                color: isSelected ? AppColors.primaryBlue : AppColors.textGray,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _SearchableListModal extends StatefulWidget {
  final String title;
  final List<String> items;
  final Function(String) onSelected;

  const _SearchableListModal({
    required this.title,
    required this.items,
    required this.onSelected,
  });

  @override
  State<_SearchableListModal> createState() => _SearchableListModalState();
}

class _SearchableListModalState extends State<_SearchableListModal> {
  late List<String> _filteredItems;
  final _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _filteredItems = widget.items;
  }

  void _filter(String query) {
    setState(() {
      _filteredItems = widget.items
          .where((item) => item.toLowerCase().contains(query.toLowerCase()))
          .toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.7,
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                widget.title,
                style: const TextStyle(color: AppColors.textWhite, fontSize: 18, fontWeight: FontWeight.bold),
              ),
              IconButton(
                icon: const Icon(Icons.close, color: AppColors.textWhite),
                onPressed: () => Navigator.pop(context),
              ),
            ],
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _searchController,
            onChanged: _filter,
            style: const TextStyle(color: AppColors.textWhite),
            decoration: InputDecoration(
              hintText: "Search...",
              hintStyle: const TextStyle(color: AppColors.textFaint),
              prefixIcon: const Icon(Icons.search, color: AppColors.textGray),
              filled: true,
              fillColor: AppColors.darkNavyCard,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: ListView.builder(
              itemCount: _filteredItems.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(_filteredItems[index], style: const TextStyle(color: AppColors.textWhite)),
                  onTap: () {
                    widget.onSelected(_filteredItems[index]);
                    Navigator.pop(context);
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

