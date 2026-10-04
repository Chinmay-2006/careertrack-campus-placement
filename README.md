# CareerTrack — Campus Placement Readiness System

CareerTrack is a student-focused machine learning application designed to help students evaluate their placement readiness through self-assessment, placement prediction, placed-student benchmarking, skill-gap analysis, personalized recommendations, and a 30-day improvement roadmap.

## 1. Project Objective

Students often have academic marks and technical skills but do not have a structured way to understand their current placement readiness, compare their profile with placement patterns, and identify the areas that require the most attention.

CareerTrack brings these capabilities together in one platform:

- Student profile and self-assessment
- Input validation
- Placement prediction
- Placement probability
- Expected starting CTC estimation
- Dataset-based benchmarking
- Skill-gap analysis
- Personalized recommendations
- 30-day improvement roadmap
- Re-assessment and assessment history

## 2. System Workflow

```text
Student Profile
       ↓
Self-Assessment
       ↓
Input Validation
       ↓
ML Prediction
       ↓
Placed-Student Benchmarking
       ↓
Skill-Gap Analysis
       ↓
Personalized Guidance
       ↓
30-Day Roadmap
       ↓
Re-assessment

##  3. Key Features
- Student profile and academic assessment
- Technical and placement skill assessment
- Numeric range and consistency validation
- Placement status prediction
- Placement probability
- Expected starting CTC prediction
- Live placed-student benchmarking
- Dataset-driven skill-gap analysis
- Personalized improvement recommendations
- 30-day improvement roadmap
- Assessment history within the current session
- Model comparison
- Cross-validation results
- Classification reports
- Confusion matrices
- Regression evaluation
- Classification feature effects
- Dataset analytics

##  4. Two-Stage Machine Learning Architecture
Placement status and starting CTC are different prediction tasks. Therefore, CareerTrack uses a two-stage machine learning architecture.

Stage 1 — Classification
The classification stage predicts:
- Placement Status
- Placement Probability
The project compares:
- Logistic Regression
- Random Forest Classifier

Stage 2 — Regression
For students predicted as placed, the regression stage estimates:
- Expected Starting CTC in LPA
The project compares:
- Linear Regression
- Random Forest Regressor

The regression model is trained using placed-student records because CTC is meaningful only for the placed population in this project.
Architecture

Student Profile
       ↓
Preprocessing Pipeline
       ↓
Stage 1 — Classification
       ↓
Placement Prediction
       ↓
If Placed
       ↓
Stage 2 — CTC Regression
       ↓
Expected Starting CTC

## 5. Dataset
The project dataset contains:
- 100,000 student records
- 26 columns
- 23 predictive input features
- 54,459 placed students
- 45,541 not-placed students
- 0 missing values
- 0 duplicate rows
The dataset contains academic, technical, communication, activity, portfolio, and profile information.

Input Features
- Age
- Gender
- CGPA
- Branch
- College Tier
- Internships Count
- Projects Count
- Certifications Count
- Coding Skill Score
- Aptitude Score
- Communication Skill Score
- Logical Reasoning Score
- Hackathons Participated
- GitHub Repositories
- LinkedIn Connections
- Mock Interview Score
- Attendance Percentage
- Backlogs
- Extracurricular Score
- Leadership Score
- Volunteer Experience
- Sleep Hours
- Study Hours Per Day

Target Variables
Classification target:
placement_status

Regression target:
salary_package_lpa

Student ID is used for assessment identification and is excluded from the predictive feature set.

##  6. Data Preprocessing
Numerical features are standardized using:
StandardScaler

Categorical features are encoded using:
OneHotEncoder

Unknown categorical values are handled through the preprocessing pipeline.
The preprocessing steps and trained estimator are stored together inside Scikit-learn pipelines so that the same transformations are applied during prediction.

##  7. Classification
The classification stage compares two models:
- Logistic Regression
- Random Forest Classifier
Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
Five-fold cross-validation is also used to evaluate model stability.
The selected classification model is determined from the evaluation results.

##  8. Regression
The regression stage compares:
- Linear Regression
- Random Forest Regressor
The regression task is performed on placed-student records.
Evaluation Metrics
- MAE
- RMSE
- R²
Five-fold cross-validation is also used to evaluate regression model stability.
The selected regression model is determined from the evaluation results.

##  9. Dataset-Driven Skill-Gap Analysis
CareerTrack does not use arbitrary requirements such as:
"Three internships are required."

Instead, improvement gaps are derived from the observed distribution of placed students in the project dataset.
The system selects the most relevant benchmark cohort using:
1. Same branch + same college tier, when enough records are available
2. Same branch, when the first cohort is too small
3. All placed students, as the final fallback
The student's current values are compared with the selected placed-student cohort using:
- Median
- Percentile
- Distribution gap
Model feature relevance can also be used as an additional prioritization signal.
The recommendation layer is separate from the machine-learning prediction model.
These benchmarks describe observed patterns in the project dataset. They are not guaranteed placement requirements and do not establish causality.

##  10. Input Validation
The application validates:
- Student name structure
- Student ID format
- Numeric ranges
- Basic logical consistency
- Values outside the observed training-data range
Examples of validation include:
- CGPA must remain within the valid range
- Skill scores must remain within their allowed range
- Attendance must remain within its valid range
- Study hours and sleep hours together cannot exceed 24 hours per day
- Student ID must follow the accepted format
The application cannot independently verify whether a student has entered truthful academic marks or assessment scores.
Therefore, these values are treated as self-reported information.

##  11. Streamlit Application
The Streamlit application contains four main sections.

--> 11.1 Student Assessment
Students enter:
- Personal details
- Academic profile
- Placement-related profile information
- Portfolio activity
- Technical assessment scores
- Placement assessment scores
The page also provides a live placed-student benchmark based on the selected branch and college tier.
After validation, the application performs the two-stage prediction.

--> 11.2 Roadmap & Progress
The student can view:
- Placement probability
- Readiness level
- Top improvement areas
- Current values
- Placed-student benchmarks
- Dataset percentile
- Personalized recommendations
- 30-day improvement roadmap
- Assessment history

--> 11.3 Model & Dataset
This section provides:
- Dataset summary
- Placement distribution
- Branch-based distribution
- College-tier distribution
- Numerical feature summary
- Two-stage ML architecture
- Preprocessing details
- Selected models
- Classification model comparison
- Classification cross-validation
- Detailed classification reports
- Confusion matrices
- Regression model comparison
- Regression cross-validation
- Regression error statistics
- Classification feature effects
- Dataset limitations

--> 11.4 About Project
This section explains:
- Problem statement
- Project objective
- System workflow
- Machine-learning approach
- Dataset-driven recommendation methodology
- Input reliability
- Technology stack
- Project modules
- Limitations
- Future scope
- Team

##  12. Model Evaluation Report
The model evaluation results are stored in:
reports/model_evaluation.json

The report contains:
- Classification performance
- Classification cross-validation
- Regression performance
- Regression cross-validation
- Regression error statistics
- Selected classification model
- Selected regression model
- Dataset information

##  13. Explainability
For linear classification, model coefficients are used to show feature effects.
For tree-based classification, feature importance is used where supported.
These values describe model behavior.
They should not be interpreted as causal relationships.

##  14. Assessment and Recommendation Flow
The complete student-facing flow is:
Student Details
       ↓
Academic Profile
       ↓
Technical Assessment
       ↓
Input Validation
       ↓
Live Benchmark
       ↓
Placement Prediction
       ↓
Expected CTC
       ↓
Skill-Gap Analysis
       ↓
Personalized Recommendations
       ↓
30-Day Improvement Roadmap
       ↓
Re-assessment

This makes CareerTrack more than a simple placement prediction system. It provides a complete placement-readiness assessment and improvement workflow.

##  15. Project Structure
careertrack-campus-placement/
│
├── data/
│   └── student_placement_prediction_dataset_2026.csv
│
├── models/
│   ├── placement_classifier.pkl
│   └── ctc_regressor.pkl
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   └── 02_model_development.ipynb
│
├── reports/
│   └── model_evaluation.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── prediction.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── validation.py
│   └── recommendation.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

##  16. Important Project Modules
config.py
Stores project configuration, feature definitions, targets, and model paths.
data_loader.py
Loads the dataset and prepares data for classification and regression.
preprocessing.py
Creates the numerical scaling and categorical encoding pipeline.
train_classification.py
Trains and compares classification models.
train_regression.py
Trains and compares regression models using placed-student records.
prediction.py
Loads trained models and performs student-level predictions.
evaluation.py
Calculates evaluation metrics and cross-validation results.
explainability.py
Provides classification feature-effect information.
validation.py
Validates student inputs and identifies data-quality issues.
recommendation.py
Performs placed-student benchmarking, skill-gap analysis, personalized recommendations, and roadmap generation.
app.py
Provides the Streamlit application interface.
##  17. Technology Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter
- Git
- GitHub

##  18. Running the Project Locally
Create and activate a virtual environment and install the dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

The application will open in the Streamlit interface.

##  19. Limitations
- The dataset is synthetic.
- Student-entered information is not independently verified.
- Dataset benchmarks represent observed patterns, not guaranteed placement requirements.
- Model associations do not establish causal relationships.
- Some real-world placement factors may not be represented in the dataset.
- Predictions, benchmarks, recommendations, and roadmaps are guidance rather than guarantees.

##  20. Future Scope
Possible future improvements include:
- Authenticated student accounts
- Persistent database storage
- Verified academic-document upload
- Progress tracking across semesters
- Company-specific preparation plans
- Integration with external assessment platforms
- Long-term student progress analytics
- More advanced model tuning and comparison

##  21. Team
- Chinmay Patil
- Sanika Mhatre
- Siddharth Parchande
- Dhruva Mhatre