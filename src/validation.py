import math
import re

import pandas as pd


# ============================================================
# NAME VALIDATION
# ============================================================

def validate_student_name(name):
    """
    Validate the structure of a student's name.

    This does NOT verify identity.
    """

    errors = []

    if not isinstance(name, str):

        errors.append(
            "Student name must be text."
        )

        return errors

    name = name.strip()

    if not name:

        errors.append(
            "Student name cannot be empty."
        )

        return errors

    if len(name) < 3:

        errors.append(
            "Student name must contain at least 3 characters."
        )

    if len(name) > 100:

        errors.append(
            "Student name is too long."
        )

    if not re.fullmatch(
        r"[A-Za-z]+(?:[ .'-][A-Za-z]+)*",
        name
    ):

        errors.append(
            "Student name contains invalid characters. "
            "Use letters, spaces, apostrophes or hyphens only."
        )

    # Reject obvious repeated-character input.
    compact = re.sub(
        r"[^A-Za-z]",
        "",
        name
    ).lower()

    if compact:

        if len(set(compact)) == 1:

            errors.append(
                "Student name does not appear to be a valid name."
            )

    return errors


# ============================================================
# STUDENT ID VALIDATION
# ============================================================

def validate_student_id(student_id):
    """
    Validate the structure of a student ID / roll number.
    """

    errors = []

    if not isinstance(
        student_id,
        str
    ):

        errors.append(
            "Student ID must be entered as text."
        )

        return errors

    student_id = student_id.strip()

    if not student_id:

        errors.append(
            "Student ID cannot be empty."
        )

        return errors

    if len(student_id) > 50:

        errors.append(
            "Student ID is too long."
        )

    if not re.fullmatch(
        r"[A-Za-z0-9/_-]+",
        student_id
    ):

        errors.append(
            "Student ID may contain only letters, numbers, "
            "slash, underscore or hyphen."
        )

    return errors


# ============================================================
# NUMERIC RANGE VALIDATION
# ============================================================

def _check_range(
    values,
    field,
    minimum,
    maximum
):
    """
    Validate one numeric value.
    """

    errors = []

    value = values.get(field)

    if value is None:
        return errors

    try:

        numeric_value = float(value)

    except (
        TypeError,
        ValueError
    ):

        errors.append(
            f"{field} must be numeric."
        )

        return errors

    if not math.isfinite(
        numeric_value
    ):

        errors.append(
            f"{field} must be a finite number."
        )

        return errors

    if numeric_value < minimum:

        errors.append(
            f"{field} cannot be less than {minimum}."
        )

    if numeric_value > maximum:

        errors.append(
            f"{field} cannot be greater than {maximum}."
        )

    return errors


# ============================================================
# DATASET PLAUSIBILITY
# ============================================================

def _dataset_range_warnings(
    student_values,
    df
):
    """
    Compare student values with the observed dataset range.

    These are warnings, not proof that an input is false.
    """

    warnings = []

    if df is None or df.empty:
        return warnings

    numeric_columns = [
        column
        for column in df.columns
        if column in student_values
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        student_value = student_values.get(
            column
        )

        if student_value is None:
            continue

        try:

            student_value = float(
                student_value
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        minimum = float(
            series.min()
        )

        maximum = float(
            series.max()
        )

        if (
            student_value < minimum
            or student_value > maximum
        ):

            warnings.append(
                f"{column}: value {student_value:g} "
                f"is outside the observed training-data range "
                f"({minimum:g} to {maximum:g})."
            )

    return warnings


# ============================================================
# CONSISTENCY VALIDATION
# ============================================================

def _consistency_checks(values):
    """
    Check combinations of values that should make sense.
    """

    errors = []

    study_hours = values.get(
        "study_hours_per_day"
    )

    sleep_hours = values.get(
        "sleep_hours_per_day"
    )

    if (
        study_hours is not None
        and sleep_hours is not None
    ):

        try:

            total = (
                float(study_hours)
                + float(sleep_hours)
            )

            if total > 24:

                errors.append(
                    "Study hours plus sleep hours cannot exceed 24 hours per day."
                )

        except (
            TypeError,
            ValueError
        ):

            pass

    return errors


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_all(
    student_values,
    df=None
):
    """
    Run all input validation checks.

    Returns:

        {
            "errors": [...],
            "warnings": [...],
            "dataset_warnings": [...]
        }
    """

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Identity validation
    # --------------------------------------------------------

    errors.extend(
        validate_student_name(
            student_values.get(
                "student_name",
                ""
            )
        )
    )

    errors.extend(
        validate_student_id(
            student_values.get(
                "student_id",
                ""
            )
        )
    )

    # --------------------------------------------------------
    # Main numeric ranges
    # --------------------------------------------------------

    range_rules = {

        "age": (
            18,
            30
        ),

        "cgpa": (
            0,
            10
        ),

        "attendance_percentage": (
            0,
            100
        ),

        "backlogs": (
            0,
            20
        ),

        "internships_count": (
            0,
            20
        ),

        "projects_count": (
            0,
            30
        ),

        "certifications_count": (
            0,
            30
        ),

        "hackathons_participated": (
            0,
            20
        ),

        "github_repos": (
            0,
            500
        ),

        "linkedin_connections": (
            0,
            5000
        ),

        "extracurricular_score": (
            0,
            100
        ),

        "leadership_score": (
            0,
            100
        ),

        "study_hours_per_day": (
            0,
            24
        ),

        "sleep_hours_per_day": (
            0,
            24
        ),

        "coding_skill_score": (
            0,
            100
        ),

        "aptitude_score": (
            0,
            100
        ),

        "communication_skill_score": (
            0,
            100
        ),

        "logical_reasoning_score": (
            0,
            100
        ),

        "mock_interview_score": (
            0,
            100
        )
    }

    for field, bounds in range_rules.items():

        errors.extend(
            _check_range(
                student_values,
                field,
                bounds[0],
                bounds[1]
            )
        )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    errors.extend(
        _consistency_checks(
            student_values
        )
    )

    # --------------------------------------------------------
    # Dataset plausibility
    # --------------------------------------------------------

    dataset_warnings = (
        _dataset_range_warnings(
            student_values,
            df
        )
    )

    warnings.extend(
        dataset_warnings
    )

    return {
        "errors": errors,
        "warnings": warnings,
        "dataset_warnings": dataset_warnings
    }