import pandas as pd


# ============================================================
# RECOMMENDATION CONFIGURATION
# ============================================================

# Higher values are considered better.
HIGHER_IS_BETTER = {
    "CGPA": {
        "column": "cgpa",
        "action": (
            "Improve CGPA through consistent academic revision "
            "and stronger performance in upcoming assessments."
        ),
        "core": True
    },

    "Attendance": {
        "column": "attendance_percentage",
        "action": (
            "Maintain strong academic attendance and improve "
            "consistency in college activities."
        ),
        "core": False
    },

    "Coding Skill": {
        "column": "coding_skill_score",
        "action": (
            "Strengthen programming fundamentals and practice "
            "DSA, Java and problem-solving questions regularly."
        ),
        "core": True
    },

    "Aptitude": {
        "column": "aptitude_score",
        "action": (
            "Practice quantitative aptitude, logical reasoning "
            "and placement-style timed questions."
        ),
        "core": True
    },

    "Communication": {
        "column": "communication_skill_score",
        "action": (
            "Improve communication through mock HR interviews, "
            "structured speaking and technical explanation practice."
        ),
        "core": True
    },

    "Logical Reasoning": {
        "column": "logical_reasoning_score",
        "action": (
            "Practice logical reasoning sets regularly and focus "
            "on speed and accuracy."
        ),
        "core": True
    },

    "Mock Interview": {
        "column": "mock_interview_score",
        "action": (
            "Take regular mock interviews and improve technical "
            "explanation, confidence and response structure."
        ),
        "core": True
    },

    "Internships": {
        "column": "internships_count",
        "action": (
            "Build practical experience through internships, "
            "industry projects or relevant practical work."
        ),
        "core": True
    },

    "Projects": {
        "column": "projects_count",
        "action": (
            "Build meaningful technical projects and document "
            "them properly on GitHub."
        ),
        "core": True
    },

    "Certifications": {
        "column": "certifications_count",
        "action": (
            "Complete relevant certifications that strengthen "
            "your technical profile."
        ),
        "core": False
    },

    "Hackathons": {
        "column": "hackathons_participated",
        "action": (
            "Participate in hackathons to improve practical "
            "problem-solving and collaborative development."
        ),
        "core": False
    },

    "GitHub Activity": {
        "column": "github_repos",
        "action": (
            "Improve your GitHub portfolio with complete projects, "
            "clear README files and meaningful commits."
        ),
        "core": False
    },

    "LinkedIn Network": {
        "column": "linkedin_connections",
        "action": (
            "Maintain a professional LinkedIn profile and build "
            "a relevant professional network."
        ),
        "core": False
    },

    "Extracurricular": {
        "column": "extracurricular_score",
        "action": (
            "Strengthen extracurricular involvement through "
            "meaningful technical, leadership or community activities."
        ),
        "core": False
    },

    "Leadership": {
        "column": "leadership_score",
        "action": (
            "Take ownership of projects, teams or activities "
            "to develop leadership experience."
        ),
        "core": False
    },

    "Study Hours": {
        "column": "study_hours_per_day",
        "action": (
            "Increase focused study time gradually while maintaining "
            "a sustainable daily routine."
        ),
        "core": False
    }
}


# Lower values are considered better.
LOWER_IS_BETTER = {
    "Backlogs": {
        "column": "backlogs",
        "action": (
            "Focus on clearing academic backlogs and maintaining "
            "a clean academic record."
        ),
        "core": True
    }
}


# ============================================================
# COHORT SELECTION
# ============================================================

def _select_placed_cohort(student_values, df):
    """
    Select the most relevant placed-student cohort.

    Priority:
        1. Same branch + same college tier
        2. Same branch
        3. All placed students

    A minimum cohort size of 30 is used.
    """

    if df is None or df.empty:
        return pd.DataFrame(), "No benchmark available"

    required_column = "placement_status"

    if required_column not in df.columns:
        return pd.DataFrame(), "No benchmark available"

    placed_df = df[
        df[required_column] == "Placed"
    ].copy()

    if placed_df.empty:
        return pd.DataFrame(), "No placed students available"

    branch = student_values.get("branch")
    college_tier = student_values.get("college_tier")

    # --------------------------------------------------------
    # Same branch + same college tier
    # --------------------------------------------------------

    same_branch_tier = placed_df[
        (placed_df["branch"] == branch)
        & (placed_df["college_tier"] == college_tier)
    ].copy()

    if len(same_branch_tier) >= 30:
        return (
            same_branch_tier,
            f"{branch} + {college_tier}"
        )

    # --------------------------------------------------------
    # Same branch
    # --------------------------------------------------------

    same_branch = placed_df[
        placed_df["branch"] == branch
    ].copy()

    if len(same_branch) >= 30:
        return (
            same_branch,
            f"{branch}"
        )

    # --------------------------------------------------------
    # All placed students
    # --------------------------------------------------------

    return (
        placed_df,
        "All placed students"
    )


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _numeric_series(df, column):
    """
    Convert a dataset column to numeric safely.
    """

    if column not in df.columns:
        return pd.Series(dtype="float64")

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return values.dropna()


def _statistics(series):
    """
    Calculate statistics for a dataset feature.
    """

    if series.empty:
        return {
            "q25": None,
            "median": None,
            "q75": None,
            "min": None,
            "max": None,
            "iqr": None
        }

    q25 = float(series.quantile(0.25))
    median = float(series.median())
    q75 = float(series.quantile(0.75))
    minimum = float(series.min())
    maximum = float(series.max())

    return {
        "q25": q25,
        "median": median,
        "q75": q75,
        "min": minimum,
        "max": maximum,
        "iqr": q75 - q25
    }


# ============================================================
# MODEL RELEVANCE
# ============================================================

def _model_relevance(classifier, column_name):
    """
    Use the selected Logistic Regression coefficient as
    an additional relevance signal.

    This does not determine the benchmark.
    """

    try:

        if classifier is None:
            return 0.0

        if not hasattr(classifier, "named_steps"):
            return 0.0

        if "model" not in classifier.named_steps:
            return 0.0

        if "preprocessor" not in classifier.named_steps:
            return 0.0

        model = classifier.named_steps["model"]

        if not hasattr(model, "coef_"):
            return 0.0

        preprocessor = (
            classifier.named_steps["preprocessor"]
        )

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        coefficients = model.coef_[0]

        matches = []

        for index, feature_name in enumerate(
            feature_names
        ):

            if feature_name.endswith(
                f"__{column_name}"
            ):

                matches.append(
                    abs(float(coefficients[index]))
                )

        if not matches:
            return 0.0

        return max(matches)

    except Exception:
        return 0.0


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def analyze_skill_gaps(
    student_values,
    df,
    classifier=None
):
    """
    Compare the student's profile with placed students.

    Higher-is-better feature:
        current < median -> improvement gap

    Lower-is-better feature:
        current > median -> improvement gap

    Returns dictionaries with one stable schema:

        Area
        Column
        Current
        Benchmark
        Median
        Q25
        Q75
        Percentile
        Gap
        Severity
        Model Relevance
        Core
        Priority
        Status
        Recommendation
        Cohort
        Priority Score
    """

    cohort, cohort_name = _select_placed_cohort(
        student_values,
        df
    )

    if cohort.empty:
        return []

    areas = {}

    areas.update(HIGHER_IS_BETTER)
    areas.update(LOWER_IS_BETTER)

    gaps = []

    # ========================================================
    # FEATURE ANALYSIS
    # ========================================================

    for area, config in areas.items():

        column = config["column"]

        series = _numeric_series(
            cohort,
            column
        )

        if series.empty:
            continue

        stats = _statistics(series)

        median = stats["median"]
        q25 = stats["q25"]
        q75 = stats["q75"]
        iqr = stats["iqr"]

        current = student_values.get(
            column
        )

        if current is None:
            continue

        try:
            current = float(current)
        except (TypeError, ValueError):
            continue

        # ----------------------------------------------------
        # HIGHER IS BETTER
        # ----------------------------------------------------

        if area in HIGHER_IS_BETTER:

            percentile = float(
                (series <= current).mean() * 100
            )

            if current >= median:
                continue

            gap_amount = (
                median - current
            )

            denominator = max(
                abs(iqr),
                1.0
            )

            severity = (
                gap_amount / denominator
            )

            severity = min(
                severity / 2.0,
                1.0
            )

            if percentile < 25:
                priority = "High"
            else:
                priority = "Medium"

        # ----------------------------------------------------
        # LOWER IS BETTER
        # ----------------------------------------------------

        else:

            # Higher percentile means worse relative position
            # for a feature such as backlogs.
            lower_better_percentile = float(
                (series < current).mean() * 100
            )

            percentile = max(
                0.0,
                100.0 - lower_better_percentile
            )

            if current <= median:
                continue

            gap_amount = (
                current - median
            )

            denominator = max(
                abs(iqr),
                1.0
            )

            severity = (
                gap_amount / denominator
            )

            severity = min(
                severity / 2.0,
                1.0
            )

            if current > q75:
                priority = "High"
            else:
                priority = "Medium"

        # ----------------------------------------------------
        # MODEL RELEVANCE
        # ----------------------------------------------------

        relevance = _model_relevance(
            classifier,
            column
        )

        gaps.append({

            "Area": area,

            "Column": column,

            "Current": current,

            # Main public benchmark field.
            "Benchmark": median,

            # Alias for internal compatibility.
            "Median": median,

            "Q25": q25,

            "Q75": q75,

            "Percentile": percentile,

            "Gap": gap_amount,

            "Severity": severity,

            "Model Relevance": relevance,

            "Core": config["core"],

            "Priority": priority,

            "Status": priority,

            "Recommendation": config["action"],

            "Cohort": cohort_name,

            "Priority Score": 0.0
        })

    if not gaps:
        return []

    # ========================================================
    # NORMALIZE MODEL RELEVANCE
    # ========================================================

    max_relevance = max(
        gap["Model Relevance"]
        for gap in gaps
    )

    for gap in gaps:

        if max_relevance > 0:

            normalized_relevance = (
                gap["Model Relevance"]
                / max_relevance
            )

        else:

            normalized_relevance = 0.0

        gap["Normalized Relevance"] = (
            normalized_relevance
        )

    # ========================================================
    # PRIORITY SCORE
    # ========================================================

    for gap in gaps:

        severity_score = gap["Severity"]

        relevance_score = (
            gap["Normalized Relevance"]
        )

        # Core placement areas receive a modest bonus.
        core_bonus = (
            0.15
            if gap["Core"]
            else 0.0
        )

        score = (
            0.70 * severity_score
            + 0.15 * relevance_score
            + core_bonus
        )

        gap["Priority Score"] = min(
            score,
            1.0
        )

    # ========================================================
    # SORT
    # ========================================================

    gaps.sort(
        key=lambda gap: (
            gap["Priority Score"],
            gap["Core"],
            gap["Severity"]
        ),
        reverse=True
    )

    # Keep the result manageable.
    return gaps[:6]


# ============================================================
# LIVE BENCHMARK
# ============================================================

def get_live_benchmark(
    student_values,
    df
):
    """
    Generate a live benchmark preview.

    The cohort is selected using:
        branch + college tier
        -> branch
        -> all placed students
    """

    cohort, cohort_name = _select_placed_cohort(
        student_values,
        df
    )

    if cohort.empty:
        return None

    required_columns = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "projects_count",
        "internships_count"
    ]

    for column in required_columns:

        if column not in cohort.columns:
            return None

    return {

        "Cohort": cohort_name,

        "Students": int(
            len(cohort)
        ),

        "Median CGPA": float(
            pd.to_numeric(
                cohort["cgpa"],
                errors="coerce"
            ).median()
        ),

        "Median Coding": float(
            pd.to_numeric(
                cohort["coding_skill_score"],
                errors="coerce"
            ).median()
        ),

        "Median Aptitude": float(
            pd.to_numeric(
                cohort["aptitude_score"],
                errors="coerce"
            ).median()
        ),

        "Median Projects": float(
            pd.to_numeric(
                cohort["projects_count"],
                errors="coerce"
            ).median()
        ),

        "Median Internships": float(
            pd.to_numeric(
                cohort["internships_count"],
                errors="coerce"
            ).median()
        )
    }


# ============================================================
# 30-DAY ROADMAP
# ============================================================

def generate_roadmap(
    skill_gaps,
    student_values=None
):
    """
    Convert top three skill gaps into a four-week roadmap.

    Week 1-3:
        Highest priority gaps

    Week 4:
        Placement practice + reassessment
    """

    roadmap = []

    for index, gap in enumerate(
        skill_gaps[:3],
        start=1
    ):

        roadmap.append({

            "Week": f"Week {index}",

            "Focus Area": gap["Area"],

            "Priority": gap["Priority"],

            "Current": gap["Current"],

            "Benchmark": gap["Benchmark"],

            "Placed-Student Median": gap["Benchmark"],

            "Percentile": gap["Percentile"],

            "Action": gap["Recommendation"],

            "Cohort": gap["Cohort"]
        })

    # --------------------------------------------------------
    # Always provide a Week 4.
    # --------------------------------------------------------

    roadmap.append({

        "Week": "Week 4",

        "Focus Area": "Placement Practice",

        "Priority": "Medium",

        "Current": None,

        "Benchmark": None,

        "Placed-Student Median": None,

        "Percentile": None,

        "Action": (
            "Complete a timed placement mock test covering "
            "aptitude, DSA and interview preparation. "
            "Record mistakes, revise weak topics and then "
            "perform a fresh assessment."
        ),

        "Cohort": None
    })

    return roadmap


# ============================================================
# READINESS LABEL
# ============================================================

def get_readiness_label(
    placement_probability
):
    """
    Convert placement probability into a simple
    readiness category.
    """

    probability = float(
        placement_probability
    )

    if probability >= 0.75:
        return "Strong"

    if probability >= 0.60:
        return "Moderate"

    return "Needs Improvement"