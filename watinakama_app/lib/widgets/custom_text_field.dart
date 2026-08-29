import 'package:flutter/material.dart';
import '../constants/app_colors.dart';

class CustomTextField extends StatefulWidget {
  final String label;
  final bool isPassword;
  final TextEditingController controller;
  final TextInputType keyboardType;
  final String? prefixText; // for +94 phone prefix
  final Widget? prefixWidget; // for dropdown prefix
  final String? hint;
  final bool enabled;
  final IconData? suffixIcon;

  const CustomTextField({
    super.key,
    required this.label,
    this.isPassword = false,
    required this.controller,
    this.keyboardType = TextInputType.text,
    this.prefixText,
    this.prefixWidget,
    this.hint,
    this.enabled = true,
    this.suffixIcon,
  });

  @override
  State<CustomTextField> createState() => _CustomTextFieldState();
}

class _CustomTextFieldState extends State<CustomTextField> {
  bool _obscureText = true;

  @override
  void initState() {
    super.initState();
    _obscureText = widget.isPassword;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Label above the field (only if provided)
        if (widget.label.isNotEmpty) ...[
          Text(
            widget.label,
            style: const TextStyle(
              color: AppColors.textGray,
              fontSize: 12,
              fontWeight: FontWeight.w400,
            ),
          ),
          const SizedBox(height: 4),
        ],
        
        // TextField Container
        Container(
          decoration: BoxDecoration(
            color: widget.enabled ? AppColors.darkNavyCard : AppColors.darkNavyCard.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: AppColors.textFieldBorder,
              width: 1,
            ),
          ),
          child: Row(
            children: [
              // Prefix Widget or Text
              if (widget.prefixWidget != null) ...[
                Padding(
                  padding: const EdgeInsets.only(left: 12),
                  child: widget.prefixWidget!,
                ),
                Container(
                  height: 20,
                  width: 1,
                  margin: const EdgeInsets.symmetric(horizontal: 10),
                  color: AppColors.textFieldBorder,
                ),
              ] else if (widget.prefixText != null) ...[
                Padding(
                  padding: const EdgeInsets.only(left: 12),
                  child: Text(
                    widget.prefixText!,
                    style: const TextStyle(
                      color: AppColors.textWhite,
                      fontSize: 14,
                    ),
                  ),
                ),
                Container(
                  height: 18,
                  width: 1,
                  margin: const EdgeInsets.symmetric(horizontal: 10),
                  color: AppColors.textFieldBorder,
                ),
              ],

              // Actual TextField
              Expanded(
                child: TextField(
                  controller: widget.controller,
                  enabled: widget.enabled,
                  obscureText: widget.isPassword ? _obscureText : false,
                  keyboardType: widget.keyboardType,
                  style: TextStyle(
                    color: widget.enabled ? AppColors.textWhite : AppColors.textGray,
                    fontSize: 14,
                  ),
                  decoration: InputDecoration(
                    hintText: widget.hint,
                    hintStyle: const TextStyle(
                      color: AppColors.textGray,
                      fontSize: 14,
                    ),
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: (widget.prefixText == null && widget.prefixWidget == null) ? 12 : 0,
                      vertical: 12,
                    ),
                    border: InputBorder.none,
                    focusedBorder: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    errorBorder: InputBorder.none,
                    disabledBorder: InputBorder.none,
                  ),
                ),
              ),

              // Suffix Icon (e.g. lock for disabled fields)
              if (widget.suffixIcon != null)
                Padding(
                  padding: const EdgeInsets.only(right: 12),
                  child: Icon(widget.suffixIcon, color: AppColors.textGray, size: 18),
                ),

              // Password Toggle Icon
              if (widget.isPassword)
                IconButton(
                  icon: Icon(
                    _obscureText ? Icons.visibility_off : Icons.visibility,
                    color: AppColors.textGray,
                    size: 20,
                  ),
                  onPressed: () {
                    setState(() {
                      _obscureText = !_obscureText;
                    });
                  },
                ),
            ],
          ),
        ),
      ],
    );
  }
}
