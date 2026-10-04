import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS
)

from src.data_loader import (
    get_dataset_summary
)

from src.prediction import (
    load_models,
    predict_student
)

from src.evaluation import (
    load_evaluation_results
)

from src.explainability import (
    get_classification_feature_effects
)

from src.recommendation import (
    analyze_skill_gaps,
    generate_roadmap,
    get_live_benchmark,
    get_readiness_label
)

from src.validation import (
    validate_all
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CareerTrack — Campus Placement Readiness System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1 {
        font-size: 2.45rem !important;
        letter-spacing: -0.7px;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.65rem !important;
        letter-spacing: -0.3px;
        margin-top: 1.4rem !important;
        margin-bottom: 0.7rem !important;
    }

    h3 {
        font-size: 1.12rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.55rem !important;
    }

    [data-testid="stCaptionContainer"] {
        opacity: 0.72;
    }

    [data-testid="stMetric"] {
        padding: 0.2rem 0;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        opacity: 0.72;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
    }

    .stButton > button {
        border-radius: 8px;
    }

    hr {
        margin-top: 1.15rem;
        margin-bottom: 1.15rem;
        opacity: 0.22;
    }

    .roadmap-week {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.62;
        margin-bottom: 0.15rem;
    }

    .roadmap-title {
        font-size: 1.22rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }

    .roadmap-priority {
        font-size: 0.86rem;
        opacity: 0.72;
        margin-bottom: 0.75rem;
    }

    .muted-note {
        font-size: 0.84rem;
        opacity: 0.68;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "CareerTrack — Campus Placement Readiness System"
)

st.caption(
    "Student-focused placement readiness analysis using "
    "machine learning, dataset-based benchmarking and "
    "personalized improvement guidance."
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():

    return load_models()


@st.cache_data
def load_project_dataset():

    return pd.read_csv(
        DATA_PATH
    )


@st.cache_data
def load_project_evaluation():

    return load_evaluation_results()


try:

    classifier, regressor = (
        load_model_artifacts()
    )

    df = load_project_dataset()

    evaluation_results = (
        load_project_evaluation()
    )

except Exception as error:

    st.error(
        f"Unable to load project artifacts: {error}"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "latest_assessment" not in st.session_state:

    st.session_state[
        "latest_assessment"
    ] = None


if "assessment_history" not in st.session_state:

    st.session_state[
        "assessment_history"
    ] = []


# ============================================================
# NAVIGATION
# ============================================================

tab_assessment, \
tab_roadmap, \
tab_model, \
tab_about = st.tabs(
    [
        "Student Assessment",
        "Roadmap & Progress",
        "Model & Dataset",
        "About Project"
    ]
)


# ============================================================
# STUDENT ASSESSMENT
# ============================================================

with tab_assessment:

    st.header(
        "Student Profile"
    )

    st.caption(
        "Enter your latest available academic, technical "
        "and profile information."
    )

    # ========================================================
    # STUDENT DETAILS
    # ========================================================

    st.subheader(
        "Student Details"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        student_name = st.text_input(
            "Student Name",
            placeholder="Enter your name"
        )

    with c2:

        student_id = st.text_input(
            "Student ID / Roll Number",
            placeholder="Enter your ID"
        )

    with c3:

        gender = st.selectbox(
            "Gender",
            sorted(
                df["gender"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with c4:

        branch = st.selectbox(
            "Branch",
            sorted(
                df["branch"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        semester = st.number_input(
            "Current Semester",
            min_value=1,
            max_value=8,
            value=7,
            step=1
        )

    with c2:

        academic_year = st.selectbox(
            "Academic Year",
            [
                "First Year",
                "Second Year",
                "Third Year",
                "Final Year"
            ],
            index=3
        )

    with c3:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=30,
            value=21,
            step=1
        )

    with c4:

        college_tier = st.selectbox(
            "College Tier",
            sorted(
                df["college_tier"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    # ========================================================
    # ACADEMIC PROFILE
    # ========================================================

    st.subheader(
        "Academic & Placement Profile"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.50,
            step=0.01
        )

    with c2:

        attendance = st.number_input(
            "Attendance %",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.01
        )

    with c3:

        backlogs = st.number_input(
            "Backlogs",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    with c4:

        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with c5:

        projects = st.number_input(
            "Projects",
            min_value=0,
            max_value=30,
            value=2,
            step=1
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=30,
            value=1,
            step=1
        )

    with c2:

        hackathons = st.number_input(
            "Hackathons",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with c3:

        volunteer = st.selectbox(
            "Volunteer Experience",
            ["No", "Yes"]
        )

    with c4:

        st.markdown(
            '<div class="muted-note">'
            "Academic/profile values are self-reported."
            "</div>",
            unsafe_allow_html=True
        )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    st.subheader(
        "Portfolio & Activity"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        github_repos = st.number_input(
            "GitHub Repositories",
            min_value=0,
            max_value=500,
            value=10,
            step=1
        )

    with c2:

        linkedin_connections = st.number_input(
            "LinkedIn Connections",
            min_value=0,
            max_value=5000,
            value=300,
            step=1
        )

    with c3:

        extracurricular = st.number_input(
            "Extracurricular Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with c4:

        leadership = st.number_input(
            "Leadership Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with c5:

        study_hours = st.number_input(
            "Study Hours / Day",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.25
        )

    # ========================================================
    # SKILLS
    # ========================================================

    st.subheader(
        "Skills & Placement Assessment"
    )

    st.caption(
        "Use your latest test, mock-assessment or honest "
        "self-assessment values."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        coding = st.number_input(
            "Coding Skill",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with c2:

        aptitude = st.number_input(
            "Aptitude",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with c3:

        communication = st.number_input(
            "Communication",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with c4:

        logical_reasoning = st.number_input(
            "Logical Reasoning",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with c5:

        mock_interview = st.number_input(
            "Mock Interview",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    c1, c2 = st.columns(2)

    with c1:

        sleep_hours = st.number_input(
            "Sleep Hours / Day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.25
        )

    with c2:

        st.caption(
            "The system checks format, ranges and consistency, "
            "but cannot independently verify self-reported values."
        )

    # ========================================================
    # MODEL INPUT
    # ========================================================

    student_values = {

        "age": age,

        "gender": gender,

        "cgpa": cgpa,

        "branch": branch,

        "college_tier": college_tier,

        "internships_count": internships,

        "projects_count": projects,

        "certifications_count": certifications,

        "coding_skill_score": coding,

        "aptitude_score": aptitude,

        "communication_skill_score": communication,

        "logical_reasoning_score": logical_reasoning,

        "hackathons_participated": hackathons,

        "github_repos": github_repos,

        "linkedin_connections": linkedin_connections,

        "mock_interview_score": mock_interview,

        "attendance_percentage": attendance,

        "backlogs": backlogs,

        "extracurricular_score": extracurricular,

        "leadership_score": leadership,

        "volunteer_experience": volunteer,

        "sleep_hours": sleep_hours,

        "study_hours_per_day": study_hours
    }

    # ========================================================
    # LIVE BENCHMARK
    # ========================================================

    st.subheader(
        "Live Placed-Student Benchmark"
    )

    st.caption(
        "The benchmark updates with the selected branch "
        "and college tier."
    )

    live_benchmark = get_live_benchmark(
        student_values,
        df
    )

    if live_benchmark is not None:

        b1, b2, b3, b4, b5, b6 = st.columns(6)

        with b1:

            st.metric(
                "Benchmark Students",
                f"{live_benchmark['Students']:,}"
            )

        with b2:

            st.metric(
                "Median CGPA",
                f"{live_benchmark['Median CGPA']:.2f}"
            )

        with b3:

            st.metric(
                "Median Coding",
                f"{live_benchmark['Median Coding']:.2f}"
            )

        with b4:

            st.metric(
                "Median Aptitude",
                f"{live_benchmark['Median Aptitude']:.2f}"
            )

        with b5:

            st.metric(
                "Median Projects",
                f"{live_benchmark['Median Projects']:.2f}"
            )

        with b6:

            st.metric(
                "Median Internships",
                f"{live_benchmark['Median Internships']:.2f}"
            )

        st.caption(
            f"Benchmark cohort: "
            f"{live_benchmark['Cohort']}"
        )

    else:

        st.info(
            "Benchmark information is currently unavailable."
        )

    # ========================================================
    # CONFIRMATION + ACTION
    # ========================================================

    st.checkbox(
        "I confirm that I have entered my latest available "
        "academic and assessment information.",
        key="assessment_confirmation"
    )

    analyze_button = st.button(
        "Analyze My Placement Readiness",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze_button:

        validation_input = dict(
            student_values
        )

        validation_input[
            "student_name"
        ] = student_name

        validation_input[
            "student_id"
        ] = student_id

        validation_input[
            "sleep_hours_per_day"
        ] = sleep_hours

        validation_result = validate_all(
            validation_input,
            df
        )

        errors = validation_result[
            "errors"
        ]

        warnings = validation_result[
            "warnings"
        ]

        # ----------------------------------------------------
        # Validation errors
        # ----------------------------------------------------

        if errors:

            st.error(
                "Please correct the following input issues."
            )

            for error in errors:

                st.write(
                    f"- {error}"
                )

            st.stop()

        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        if not st.session_state[
            "assessment_confirmation"
        ]:

            st.warning(
                "Please confirm that you have entered "
                "your latest available information."
            )

            st.stop()

        # ----------------------------------------------------
        # Dataset warnings
        # ----------------------------------------------------

        if warnings:

            with st.expander(
                "Dataset Plausibility Warnings"
            ):

                st.caption(
                    "These warnings only indicate that a value is "
                    "unusual relative to the training dataset. "
                    "They do not prove that the value is false."
                )

                for warning in warnings:

                    st.write(
                        f"- {warning}"
                    )

        # ----------------------------------------------------
        # Model dataframe
        # ----------------------------------------------------

        model_row = {
            feature: student_values[feature]
            for feature in FEATURE_COLUMNS
        }

        student_df = pd.DataFrame(
            [model_row]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        try:

            prediction = predict_student(
                student_df,
                classifier,
                regressor
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()

        # ----------------------------------------------------
        # Readiness
        # ----------------------------------------------------

        placement_probability = float(
            prediction[
                "placement_probability"
            ]
        )

        readiness = get_readiness_label(
            placement_probability
        )

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        skill_gaps = analyze_skill_gaps(
            student_values,
            df,
            classifier
        )

        roadmap = generate_roadmap(
            skill_gaps,
            student_values
        )

        # ----------------------------------------------------
        # Assessment record
        # ----------------------------------------------------

        assessment_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

        assessment = {

            "student_name":
                student_name.strip(),

            "student_id":
                student_id.strip(),

            "branch":
                branch,

            "semester":
                int(semester),

            "academic_year":
                academic_year,

            "placement_prediction":
                prediction[
                    "placement_prediction"
                ],

            "placement_probability":
                placement_probability,

            "predicted_ctc":
                prediction[
                    "predicted_ctc"
                ],

            "readiness":
                readiness,

            "skill_gaps":
                skill_gaps,

            "roadmap":
                roadmap,

            "assessment_date":
                assessment_time
        }

        st.session_state[
            "latest_assessment"
        ] = assessment

        st.session_state[
            "assessment_history"
        ].insert(
            0,
            assessment
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header(
            "Your Placement Analysis"
        )

        st.caption(
            f"Analysis completed for "
            f"{student_name.strip()} "
            f"({student_id.strip()})."
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:

            st.metric(
                "Placement Prediction",
                prediction[
                    "placement_prediction"
                ]
            )

        with r2:

            st.metric(
                "Placement Probability",
                f"{placement_probability * 100:.1f}%"
            )

        with r3:

            predicted_ctc = prediction[
                "predicted_ctc"
            ]

            ctc_value = (
                "N/A"
                if predicted_ctc is None
                else f"{predicted_ctc:.2f} LPA"
            )

            st.metric(
                "Expected CTC",
                ctc_value
            )

        with r4:

            st.metric(
                "Readiness",
                readiness
            )

        st.info(
            f"Current readiness assessment: {readiness}"
        )

        st.caption(
            "Placement and CTC values are model estimates, "
            "not guarantees."
        )

        # ====================================================
        # PROFILE SUMMARY
        # ====================================================

        st.subheader(
            "Assessment Profile"
        )

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:

            st.caption("Student")
            st.write(
                student_name.strip()
            )

        with p2:

            st.caption("Student ID")
            st.write(
                student_id.strip()
            )

        with p3:

            st.caption("Branch")
            st.write(
                branch
            )

        with p4:

            st.caption("Semester")
            st.write(
                semester
            )

        with p5:

            st.caption("Academic Year")
            st.write(
                academic_year
            )

        # ====================================================
        # IMPROVEMENT PREVIEW
        # ====================================================

        st.subheader(
            "Top Improvement Areas"
        )

        if skill_gaps:

            for gap in skill_gaps[:3]:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**{gap['Area']}**  "
                        f"· {gap['Priority']} Priority"
                    )

                    g1, g2, g3 = st.columns(3)

                    with g1:

                        st.caption("Your Value")

                        st.write(
                            f"{gap['Current']:.2f}"
                        )

                    with g2:

                        st.caption(
                            "Placed-Student Benchmark"
                        )

                        st.write(
                            f"{gap['Benchmark']:.2f}"
                        )

                    with g3:

                        st.caption(
                            "Dataset Percentile"
                        )

                        st.write(
                            f"{gap['Percentile']:.1f}%"
                        )

                    st.write(
                        gap["Recommendation"]
                    )

                    st.caption(
                        f"Benchmark cohort: {gap['Cohort']}"
                    )

        else:

            st.success(
                "No major improvement gap was detected "
                "relative to the selected placed-student benchmark."
            )


# ============================================================
# ROADMAP & PROGRESS
# ============================================================

with tab_roadmap:

    st.header(
        "Roadmap & Progress"
    )

    assessment = st.session_state[
        "latest_assessment"
    ]

    if assessment is None:

        st.info(
            "Complete a Student Assessment to generate "
            "your personalized roadmap."
        )

    else:

        st.subheader(
            f"Personalized Roadmap for "
            f"{assessment['student_name']}"
        )

        st.caption(
            f"Assessment date: "
            f"{assessment['assessment_date']}  ·  "
            f"Branch: {assessment['branch']}  ·  "
            f"Semester: {assessment['semester']}"
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Placement Probability",
                f"{assessment['placement_probability'] * 100:.1f}%"
            )

        with m2:

            st.metric(
                "Improvement Areas",
                len(
                    assessment["skill_gaps"]
                )
            )

        with m3:

            st.metric(
                "Readiness",
                assessment["readiness"]
            )

        st.caption(
            "Benchmarks are calculated from placed students "
            "in the project dataset using the most relevant "
            "available cohort."
        )

        # ====================================================
        # TOP AREAS
        # ====================================================

        st.subheader(
            "Top Areas to Improve"
        )

        gaps = assessment[
            "skill_gaps"
        ]

        if not gaps:

            st.success(
                "No major improvement gaps were identified."
            )

        else:

            for gap in gaps[:3]:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### {gap['Area']}"
                    )

                    st.caption(
                        f"{gap['Priority']} Priority"
                    )

                    g1, g2, g3 = st.columns(3)

                    with g1:

                        st.caption(
                            "Your Value"
                        )

                        st.write(
                            f"{gap['Current']:.2f}"
                        )

                    with g2:

                        st.caption(
                            "Placed-Student Benchmark"
                        )

                        st.write(
                            f"{gap['Benchmark']:.2f}"
                        )

                    with g3:

                        st.caption(
                            "Dataset Percentile"
                        )

                        st.write(
                            f"{gap['Percentile']:.1f}%"
                        )

                    st.write(
                        gap["Recommendation"]
                    )

                    st.caption(
                        f"Benchmark cohort: "
                        f"{gap['Cohort']}"
                    )

        # ====================================================
        # ROADMAP
        # ====================================================

        st.subheader(
            "30-Day Improvement Roadmap"
        )

        for item in assessment["roadmap"]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f'<div class="roadmap-week">'
                    f'{item["Week"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="roadmap-title">'
                    f'{item["Focus Area"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="roadmap-priority">'
                    f'Priority: {item["Priority"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    item["Action"]
                )

                if item["Benchmark"] is not None:

                    st.caption(
                        f"Current: {item['Current']:.2f}  ·  "
                        f"Benchmark: {item['Benchmark']:.2f}  ·  "
                        f"Percentile: {item['Percentile']:.1f}%"
                    )

        # ====================================================
        # METHODOLOGY
        # ====================================================

        with st.expander(
            "How Recommendations Are Generated"
        ):

            st.write(
                "The recommendation layer compares the student's "
                "current profile with observed placed-student "
                "records in the project dataset."
            )

            st.write(
                "The system first attempts to use a sufficiently "
                "large same-branch and same-college-tier cohort. "
                "If that cohort is too small, it falls back to "
                "the same branch and finally to all placed students."
            )

            st.write(
                "The median and percentile are used to identify "
                "relative improvement gaps. Model feature relevance "
                "is used only as an additional prioritization signal."
            )

            st.write(
                "Observed benchmarks are not guaranteed placement "
                "requirements and do not establish causality."
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        roadmap_json = json.dumps(
            assessment["roadmap"],
            indent=4
        )

        st.download_button(
            "Download 30-Day Roadmap",
            data=roadmap_json,
            file_name="30_day_placement_roadmap.json",
            mime="application/json"
        )

        # ====================================================
        # HISTORY
        # ====================================================

        st.subheader(
            "Assessment History"
        )

        history = st.session_state[
            "assessment_history"
        ]

        if history:

            history_rows = []

            for item in history:

                history_rows.append({

                    "Date":
                        item["assessment_date"],

                    "Student":
                        item["student_name"],

                    "Student ID":
                        item["student_id"],

                    "Placement":
                        item["placement_prediction"],

                    "Probability":
                        f"{item['placement_probability'] * 100:.1f}%",

                    "Expected CTC":
                        (
                            "N/A"
                            if item["predicted_ctc"] is None
                            else f"{item['predicted_ctc']:.2f} LPA"
                        ),

                    "Readiness":
                        item["readiness"]
                })

            st.dataframe(
                pd.DataFrame(
                    history_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        if st.button(
            "Clear Assessment History"
        ):

            st.session_state[
                "assessment_history"
            ] = []

            st.session_state[
                "latest_assessment"
            ] = None

            st.rerun()


# ============================================================
# MODEL & DATASET
# ============================================================

with tab_model:

    st.header(
        "Model & Dataset"
    )

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader(
        "Dataset Overview"
    )

    summary = get_dataset_summary(
        df
    )

    d1, d2, d3, d4, d5 = st.columns(5)

    with d1:

        st.metric(
            "Total Records",
            f"{summary['total_records']:,}"
        )

    with d2:

        st.metric(
            "Input Features",
            summary["total_features"]
        )

    with d3:

        st.metric(
            "Placed Records",
            f"{summary['placed_records']:,}"
        )

    with d4:

        st.metric(
            "Not Placed",
            f"{summary['not_placed_records']:,}"
        )

    with d5:

        st.metric(
            "Missing Values",
            summary["missing_values"]
        )

    st.caption(
        f"Duplicate rows: {summary['duplicate_rows']:,}"
    )

    total_records = summary[
        "total_records"
    ]

    placed_records = summary[
        "placed_records"
    ]

    if total_records > 0:

        st.metric(
            "Overall Placement Rate",
            f"{placed_records / total_records * 100:.2f}%"
        )

    # ========================================================
    # DATA DISTRIBUTION
    # ========================================================

    st.subheader(
        "Placement Distribution"
    )

    dist1, dist2 = st.columns(2)

    with dist1:

        st.caption(
            "By Branch"
        )

        branch_distribution = pd.crosstab(
            df["branch"],
            df["placement_status"]
        )

        st.dataframe(
            branch_distribution,
            use_container_width=True
        )

    with dist2:

        st.caption(
            "By College Tier"
        )

        tier_distribution = pd.crosstab(
            df["college_tier"],
            df["placement_status"]
        )

        st.dataframe(
            tier_distribution,
            use_container_width=True
        )

    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    with st.expander(
        "Numerical Feature Summary"
    ):

        numeric_df = df[
            FEATURE_COLUMNS
        ].select_dtypes(
            include="number"
        )

        if not numeric_df.empty:

            st.dataframe(
                numeric_df.describe().T,
                use_container_width=True
            )

    # ========================================================
    # ML ARCHITECTURE
    # ========================================================

    st.subheader(
        "Two-Stage ML Architecture"
    )

    st.write(
        "Student Profile → Preprocessing → "
        "Stage 1 Classification → Placement Prediction → "
        "Stage 2 Regression → Expected CTC"
    )

    st.caption(
        "Stage 1 predicts placement status. Stage 2 estimates "
        "CTC only when the classification stage predicts the "
        "student as placed."
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    st.subheader(
        "Preprocessing"
    )

    pc1, pc2 = st.columns(2)

    with pc1:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Numerical Features**"
            )

            st.write(
                "StandardScaler is used to standardize numerical inputs."
            )

    with pc2:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Categorical Features**"
            )

            st.write(
                "OneHotEncoder converts categorical values and "
                "handles unknown categories."
            )

    st.caption(
        "Preprocessing and the trained model are stored together "
        "inside Scikit-learn pipelines."
    )

    # ========================================================
    # SELECTED MODELS
    # ========================================================

    st.subheader(
        "Selected Models"
    )

    selected_classification = (
        evaluation_results.get(
            "selected_classification_model",
            "Not available"
        )
    )

    selected_regression = (
        evaluation_results.get(
            "selected_regression_model",
            "Not available"
        )
    )

    sm1, sm2 = st.columns(2)

    with sm1:

        with st.container(
            border=True
        ):

            st.metric(
                "Classification",
                selected_classification
            )

    with sm2:

        with st.container(
            border=True
        ):

            st.metric(
                "Regression",
                selected_regression
            )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    classification_results = (
        evaluation_results.get(
            "classification",
            {}
        )
    )

    if classification_results:

        st.subheader(
            "Classification Model Comparison"
        )

        rows = []

        for model_name, values in (
            classification_results.items()
        ):

            rows.append({

                "Model":
                    model_name,

                "Accuracy":
                    round(
                        values.get(
                            "accuracy",
                            0
                        ),
                        4
                    ),

                "Precision":
                    round(
                        values.get(
                            "precision",
                            0
                        ),
                        4
                    ),

                "Recall":
                    round(
                        values.get(
                            "recall",
                            0
                        ),
                        4
                    ),

                "F1 Score":
                    round(
                        values.get(
                            "f1_score",
                            0
                        ),
                        4
                    ),

                "ROC-AUC":
                    round(
                        values.get(
                            "roc_auc",
                            0
                        ),
                        4
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CLASSIFICATION CROSS VALIDATION
    # ========================================================

    classification_cv = (
        evaluation_results.get(
            "classification_cross_validation",
            {}
        )
    )

    if classification_cv:

        st.subheader(
            "Classification Cross-Validation"
        )

        rows = []

        for model_name, values in (
            classification_cv.items()
        ):

            rows.append({

                "Model":
                    model_name,

                "Accuracy Mean":
                    round(
                        values.get(
                            "accuracy_mean",
                            0
                        ),
                        4
                    ),

                "Accuracy Std":
                    round(
                        values.get(
                            "accuracy_std",
                            0
                        ),
                        4
                    ),

                "F1 Mean":
                    round(
                        values.get(
                            "f1_mean",
                            0
                        ),
                        4
                    ),

                "F1 Std":
                    round(
                        values.get(
                            "f1_std",
                            0
                        ),
                        4
                    ),

                "ROC-AUC Mean":
                    round(
                        values.get(
                            "roc_auc_mean",
                            0
                        ),
                        4
                    ),

                "ROC-AUC Std":
                    round(
                        values.get(
                            "roc_auc_std",
                            0
                        ),
                        4
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CLASSIFICATION DETAILS
    # ========================================================

    if classification_results:

        with st.expander(
            "Detailed Classification Reports"
        ):

            for model_name, values in (
                classification_results.items()
            ):

                st.markdown(
                    f"**{model_name}**"
                )

                report = values.get(
                    "classification_report",
                    {}
                )

                report_rows = []

                for label, metrics in (
                    report.items()
                ):

                    if not isinstance(
                        metrics,
                        dict
                    ):
                        continue

                    report_rows.append({

                        "Class":
                            label,

                        "Precision":
                            metrics.get(
                                "precision"
                            ),

                        "Recall":
                            metrics.get(
                                "recall"
                            ),

                        "F1":
                            metrics.get(
                                "f1-score"
                            ),

                        "Support":
                            metrics.get(
                                "support"
                            )
                    })

                if report_rows:

                    st.dataframe(
                        pd.DataFrame(
                            report_rows
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

    if classification_results:

        with st.expander(
            "Confusion Matrices"
        ):

            for model_name, values in (
                classification_results.items()
            ):

                matrix = values.get(
                    "confusion_matrix"
                )

                if matrix is None:
                    continue

                st.markdown(
                    f"**{model_name}**"
                )

                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=[
                            "Actual Not Placed",
                            "Actual Placed"
                        ],
                        columns=[
                            "Predicted Not Placed",
                            "Predicted Placed"
                        ]
                    ),
                    use_container_width=True
                )

    # ========================================================
    # REGRESSION
    # ========================================================

    regression_results = (
        evaluation_results.get(
            "regression",
            {}
        )
    )

    if regression_results:

        st.subheader(
            "Regression Model Comparison"
        )

        rows = []

        for model_name, values in (
            regression_results.items()
        ):

            rows.append({

                "Model":
                    model_name,

                "MAE (LPA)":
                    round(
                        values.get(
                            "mae",
                            0
                        ),
                        4
                    ),

                "RMSE (LPA)":
                    round(
                        values.get(
                            "rmse",
                            0
                        ),
                        4
                    ),

                "R²":
                    round(
                        values.get(
                            "r2",
                            0
                        ),
                        4
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # REGRESSION CROSS VALIDATION
    # ========================================================

    regression_cv = (
        evaluation_results.get(
            "regression_cross_validation",
            {}
        )
    )

    if regression_cv:

        st.subheader(
            "Regression Cross-Validation"
        )

        rows = []

        for model_name, values in (
            regression_cv.items()
        ):

            rows.append({

                "Model":
                    model_name,

                "MAE Mean":
                    round(
                        values.get(
                            "mae_mean",
                            0
                        ),
                        4
                    ),

                "MAE Std":
                    round(
                        values.get(
                            "mae_std",
                            0
                        ),
                        4
                    ),

                "RMSE Mean":
                    round(
                        values.get(
                            "rmse_mean",
                            0
                        ),
                        4
                    ),

                "RMSE Std":
                    round(
                        values.get(
                            "rmse_std",
                            0
                        ),
                        4
                    ),

                "R² Mean":
                    round(
                        values.get(
                            "r2_mean",
                            0
                        ),
                        4
                    ),

                "R² Std":
                    round(
                        values.get(
                            "r2_std",
                            0
                        ),
                        4
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # ADDITIONAL DETAILS
    # ========================================================

    with st.expander(
        "Regression Error Statistics"
    ):

        regression_error_stats = (
            evaluation_results.get(
                "regression_error_statistics",
                {}
            )
        )

        if regression_error_stats:

            st.json(
                regression_error_stats
            )

        else:

            st.info(
                "Regression error statistics are unavailable."
            )

    # ========================================================
    # FEATURE EFFECTS
    # ========================================================

    st.subheader(
        "Classification Feature Effects"
    )

    try:

        effects = (
            get_classification_feature_effects(
                classifier,
                top_n=15
            )
        )

        if effects is not None and not effects.empty:

            st.dataframe(
                effects,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Feature-effect information is unavailable."
            )

    except Exception as error:

        st.info(
            f"Feature-effect information unavailable: {error}"
        )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    with st.expander(
        "Dataset Limitations"
    ):

        st.write(
            "The project dataset is synthetic."
        )

        st.write(
            "Student-entered information is not independently verified."
        )

        st.write(
            "Dataset benchmarks describe observed patterns and "
            "should not be interpreted as guaranteed placement requirements."
        )

        st.write(
            "Model associations do not establish causality."
        )

        st.write(
            "Placement predictions and CTC estimates are statistical "
            "model estimates rather than guarantees."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

with tab_about:

    st.header(
        "About Project"
    )

    st.caption(
        "System overview, methodology, limitations and project scope."
    )

    # ========================================================
    # PROBLEM
    # ========================================================

    st.subheader(
        "Problem Statement"
    )

    st.write(
        "Students often have academic marks and technical skills "
        "but do not have a structured way to understand their "
        "current placement readiness, compare their profile with "
        "placement patterns and identify the areas requiring the "
        "most attention."
    )

    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.subheader(
        "Project Objective"
    )

    st.write(
        "The objective is to provide a single platform that combines "
        "placement prediction, expected CTC estimation, dataset-based "
        "benchmarking, skill-gap identification and personalized "
        "improvement guidance."
    )

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.subheader(
        "System Workflow"
    )

    with st.container(
        border=True
    ):

        st.write(
            "Student Profile"
        )

        st.write(
            "↓"
        )

        st.write(
            "Self-Assessment"
        )

        st.write(
            "↓"
        )

        st.write(
            "Input Validation"
        )

        st.write(
            "↓"
        )

        st.write(
            "ML Prediction"
        )

        st.write(
            "↓"
        )

        st.write(
            "Dataset Benchmarking"
        )

        st.write(
            "↓"
        )

        st.write(
            "Skill Gap Analysis"
        )

        st.write(
            "↓"
        )

        st.write(
            "Personalized Guidance"
        )

        st.write(
            "↓"
        )

        st.write(
            "30-Day Roadmap"
        )

        st.write(
            "↓"
        )

        st.write(
            "Re-assessment"
        )

    # ========================================================
    # ML
    # ========================================================

    st.subheader(
        "How the Machine Learning Works"
    )

    ml1, ml2 = st.columns(2)

    with ml1:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Stage 1 — Classification**"
            )

            st.write(
                "Predicts the placement category and placement probability."
            )

    with ml2:

        with st.container(
            border=True
        ):

            st.markdown(
                "**Stage 2 — Regression**"
            )

            st.write(
                "Estimates expected starting CTC for students predicted as placed."
            )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "Dataset-Driven Recommendations"
    )

    st.write(
        "Recommendations are not based on arbitrary requirements "
        "such as a fixed number of internships or projects."
    )

    st.write(
        "The system derives benchmarks from placed students in "
        "the project dataset. It prefers a sufficiently large "
        "same-branch and same-college-tier cohort, then falls back "
        "to the branch or complete placed population."
    )

    # ========================================================
    # STUDENT FOCUS
    # ========================================================

    st.subheader(
        "What Makes It Student-Focused?"
    )

    st.write(
        "The output is not limited to a placement prediction. "
        "Each assessment is associated with the student's identity "
        "and followed by placement analysis, benchmark comparison, "
        "specific improvement areas, recommendations and a roadmap."
    )

    # ========================================================
    # INPUT RELIABILITY
    # ========================================================

    st.subheader(
        "Input Reliability"
    )

    st.write(
        "The system validates name structure, Student ID format, "
        "numeric ranges, logical consistency and values outside "
        "the observed training-data range."
    )

    st.write(
        "The system cannot independently verify whether "
        "self-reported academic marks or assessment scores "
        "are truthful."
    )

    # ========================================================
    # TECHNOLOGY
    # ========================================================

    st.subheader(
        "Technology Stack"
    )

    tech1, tech2, tech3, tech4, tech5 = st.columns(5)

    with tech1:
        st.write("Python")

    with tech2:
        st.write("Pandas")

    with tech3:
        st.write("NumPy")

    with tech4:
        st.write("Scikit-learn")

    with tech5:
        st.write("Streamlit")

    # ========================================================
    # MODULES
    # ========================================================

    st.subheader(
        "Project Modules"
    )

    modules = {

        "config.py":
            "Project configuration and feature definitions.",

        "data_loader.py":
            "Dataset loading and preparation.",

        "preprocessing.py":
            "Numerical scaling and categorical encoding.",

        "train_classification.py":
            "Classification model training.",

        "train_regression.py":
            "Regression model training.",

        "prediction.py":
            "Model loading and student prediction.",

        "evaluation.py":
            "Model evaluation and cross-validation.",

        "explainability.py":
            "Classification feature effects.",

        "validation.py":
            "Input and data-quality validation.",

        "recommendation.py":
            "Dataset-driven skill-gap analysis and roadmap.",

        "app.py":
            "Streamlit application interface."
    }

    for module, description in modules.items():

        st.write(
            f"**{module}** — {description}"
        )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.subheader(
        "Limitations"
    )

    limitations = [

        "The dataset is synthetic.",

        "Student-entered information is not independently verified.",

        "Dataset benchmarks describe observed patterns, "
        "not guaranteed placement requirements.",

        "Model associations do not establish causality.",

        "Real-world placement factors may not exist in the dataset.",

        "Predictions and recommendations are guidance rather than guarantees."
    ]

    for limitation in limitations:

        st.write(
            f"- {limitation}"
        )

    # ========================================================
    # FUTURE SCOPE
    # ========================================================

    st.subheader(
        "Future Scope"
    )

    st.write(
        "Future versions can add authenticated student accounts, "
        "persistent database storage, verified academic-document "
        "upload, progress tracking across semesters, company-specific "
        "preparation plans and integration with external assessment platforms."
    )

    # ========================================================
    # TEAM
    # ========================================================

    st.subheader(
        "Team"
    )

    team1, team2, team3, team4 = st.columns(4)

    with team1:
        st.write("Chinmay Patil")

    with team2:
        st.write("Sanika Mhatre")

    with team3:
        st.write("Siddharth Parchande")

    with team4:
        st.write("Dhruva Mhatre")