def get_icd10_code_from_text(result_text: str, has_macular_edema: int, eye: str) -> str:
    text = result_text.lower()

    # Base code selection
    if "no dr" in text:
        return "E11.9"  # No laterality for no complications

    elif "mild" in text:
        base = "E11.321" if has_macular_edema else "E11.329"

    elif "moderate" in text:
        base = "E11.331" if has_macular_edema else "E11.339"

    elif "severe" in text:
        base = "E11.341" if has_macular_edema else "E11.349"

    elif "proliferative" in text:
        base = "E11.351" if has_macular_edema else "E11.359"

    else:
        return "Unknown – Unable to assign ICD-10 code"

    # Laterality suffix
    if eye.lower() == "right":
        suffix = "1"
    elif eye.lower() == "left":
        suffix = "2"

    return f"{base}{suffix}"


def describe_icd10_code(code: str) -> str:
    if code == "E11.9":
        return "Type 2 diabetes mellitus without complications"

    base_code = code[:7]  # First 7 characters like E11.331
    eye_digit = code[-1]  # Last digit: 1=right, 2=left, 3=both, 9=unspecified

    # Mapping for DR type and edema
    dr_mapping = {
        "E11.321": "mild nonproliferative diabetic retinopathy with macular edema",
        "E11.329": "mild nonproliferative diabetic retinopathy without macular edema",
        "E11.331": "moderate nonproliferative diabetic retinopathy with macular edema",
        "E11.339": "moderate nonproliferative diabetic retinopathy without macular edema",
        "E11.341": "severe nonproliferative diabetic retinopathy with macular edema",
        "E11.349": "severe nonproliferative diabetic retinopathy without macular edema",
        "E11.351": "proliferative diabetic retinopathy with macular edema",
        "E11.359": "proliferative diabetic retinopathy without macular edema"
    }

    eye_mapping = {
        "1": "right eye",
        "2": "left eye",
        "3": "both eyes",
        "9": "unspecified eye"
    }

    if base_code in dr_mapping and eye_digit in eye_mapping:
        return f"Type 2 diabetes mellitus with {dr_mapping[base_code]}, {eye_mapping[eye_digit]}"
    else:
        return "Unknown or unsupported ICD-10 code"
