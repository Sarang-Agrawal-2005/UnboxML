from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split

import io, zipfile, textwrap
from flask import send_file
from jinja2 import Template

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR




app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return ":D UnboxML Backend is running"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Get file size from the saved file
    file_size = os.path.getsize(file_path)
    
    df = pd.read_csv(file_path)
    preview = df.head(5).to_dict(orient='records')
    
    return jsonify({
        'filename': file.filename,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'preview': preview,
        'size': file_size  # Add this line
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(path)

    analysis = {
        "unique_counts": df.nunique().to_dict(),
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe_numeric": df.describe().fillna("").to_dict(),
        "describe_categorical": df.describe(include=['object']).fillna("").to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "numeric_features": df.select_dtypes(include='number').columns.tolist(),
        "categorical_features": df.select_dtypes(include='object').columns.tolist()
    }
    # Encode categorical columns for correlation
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        try:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        except:
            df_encoded[col] = -1  # fallback if encoding fails

    corr = df_encoded.corr().round(2).fillna(0)

    analysis["corr_matrix"]   = corr.values.tolist()      # 2‑D list  (heatmap data)
    analysis["corr_columns"]  = corr.columns.tolist()     # column/row labels

    return jsonify(analysis)

@app.route('/column_analysis', methods=['POST'])
def column_analysis():
    payload   = request.get_json(force=True)
    filename  = payload.get("filename")
    column    = payload.get("column")

    if not filename or not column:
        return jsonify(error="filename / column missing"), 400

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify(error="file not found"), 404

    df      = pd.read_csv(path)
    if column not in df.columns:
        return jsonify(error="column not in dataset"), 400

    s       = df[column].dropna()
    numeric = pd.api.types.is_numeric_dtype(s)

    if numeric:
        counts, bins = np.histogram(s, bins="auto")
        return jsonify({
            "dtype" : "numeric",
            "hist"  : {"bins": bins.tolist(), "counts": counts.tolist()},
            "stats" : s.describe().round(3).to_dict()
        })
    else:
        vc = s.value_counts().head(20)
        
        stats = {
            "count" : int(s.count()),
            "unique": int(s.nunique()),
            "top"   : s.mode().iloc[0] if not s.mode().empty else None,
            "freq"  : int(vc.iloc[0]) if not vc.empty else 0
        }

        return jsonify({
            "dtype"   : "categorical",
            "counts"  : vc.to_dict(),
            "stats"   : stats
        })
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer, KNNImputer
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    IterativeImputer = None

from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, 
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, 
    chi2, mutual_info_classif, mutual_info_regression
)
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def validate_and_clean_data(df):
    """Validate and clean data before processing"""
    # Remove completely empty rows and columns
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    df = df.dropna(axis=1, how='all')  # Remove columns where all values are NaN
    
    # Handle problematic characters in string columns
    for col in df.select_dtypes(include=['object']).columns:
        # Replace problematic characters
        df[col] = df[col].astype(str).str.replace(r'[^\w\s-]', '', regex=True)
        # Handle empty strings
        df[col] = df[col].replace('', 'Unknown')
        df[col] = df[col].replace('nan', 'Unknown')
    
    return df


@app.route('/preprocess_enhanced', methods=['POST'])
def preprocess_enhanced():
    """Enhanced preprocessing with comprehensive options"""
    try:
        data = request.get_json()
        
        # Add input validation
        if not data:
            return jsonify(error="No data received"), 400
            
        filename = data.get("filename")
        target = data.get("target")
        
        if not filename or not target:
            return jsonify(error="filename or target missing"), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify(error="file not found"), 404

        # Add file reading validation
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return jsonify(error=f"Failed to read CSV file: {str(e)}"), 400
            
        # Validate DataFrame
        if df.empty:
            return jsonify(error="Dataset is empty"), 400
        original_shape = df.shape
        df = validate_and_clean_data(df)
        applied_steps = []

        # Validate target column exists
        if target not in df.columns:
            return jsonify(error=f"Target column '{target}' not found in dataset"), 400

        # 1. MISSING VALUE HANDLING
        missing_method = data.get("missing_value_method", "drop")
        if missing_method != "drop" and df.isnull().sum().sum() > 0:
            if missing_method == "mean":
                # Mean for numeric, mode for categorical
                for col in df.columns:
                    if col != target:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            mode_val = df[col].mode()
                            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                            df[col].fillna(fill_val, inplace=True)
                applied_steps.append("Mean/Mode imputation applied")
                
            elif missing_method == "median":
                for col in df.columns:
                    if col != target and df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                applied_steps.append("Median imputation applied")
                
            elif missing_method == "most_frequent":
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                df = df_imputed
                applied_steps.append("Most frequent imputation applied")
                
            elif missing_method == "constant":
                constant_val = data.get("constant_value", 0)
                df.fillna(constant_val, inplace=True)
                applied_steps.append(f"Constant value imputation ({constant_val}) applied")
                
            elif missing_method == "knn":
                n_neighbors = data.get("knn_neighbors", 5)
                # Only apply to numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    applied_steps.append(f"KNN imputation (k={n_neighbors}) applied to numeric features")
                    
            elif missing_method == "iterative" and IterativeImputer:
                # Only apply to numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    applied_steps.append("Iterative imputation applied to numeric features")
        
        elif missing_method == "drop":
            df = df.dropna()
            applied_steps.append("Rows with missing values dropped")

        # 2. DUPLICATE HANDLING
        duplicate_method = data.get("duplicate_method", "none")
        if duplicate_method == "drop_all":
            df = df.drop_duplicates()
            applied_steps.append("All duplicate rows removed")
        elif duplicate_method == "keep_first":
            df = df.drop_duplicates(keep='first')
            applied_steps.append("Duplicate rows removed (kept first occurrence)")
        elif duplicate_method == "keep_last":
            df = df.drop_duplicates(keep='last')
            applied_steps.append("Duplicate rows removed (kept last occurrence)")

        # 3. OUTLIER REMOVAL
        outlier_method = data.get("outlier_method", "none")
        if outlier_method == "iqr":
            threshold = data.get("outlier_threshold", 1.5)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            applied_steps.append(f"IQR outlier removal applied (threshold={threshold})")
            
        elif outlier_method == "zscore":
            threshold = data.get("zscore_threshold", 3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
            applied_steps.append(f"Z-score outlier removal applied (threshold={threshold})")
            
        elif outlier_method == "isolation_forest":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            
            if len(numeric_cols) > 0:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(df[numeric_cols])
                df = df[outlier_labels == 1]
                applied_steps.append("Isolation Forest outlier removal applied")

        # 4. ENHANCED ENCODING WITH ERROR HANDLING
        encoding_method = data.get("encoding_method", "label")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target in categorical_cols:
            categorical_cols.remove(target)

        if len(categorical_cols) > 0:
            try:
                if encoding_method == "label":
                    le = LabelEncoder()
                    for col in categorical_cols:
                        # Handle missing values before encoding
                        df[col] = df[col].fillna('Unknown')
                        df[col] = le.fit_transform(df[col].astype(str))
                    applied_steps.append("Label encoding applied to categorical features")
                    
                elif encoding_method == "onehot":
                    # Handle missing values before one-hot encoding
                    for col in categorical_cols:
                        df[col] = df[col].fillna('Unknown')
                    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
                    applied_steps.append("One-hot encoding applied to categorical features")
                    
                elif encoding_method == "ordinal":
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    # Handle missing values before encoding
                    for col in categorical_cols:
                        df[col] = df[col].fillna('Unknown')
                    df[categorical_cols] = oe.fit_transform(df[categorical_cols].astype(str))
                    applied_steps.append("Ordinal encoding applied to categorical features")
                    
                elif encoding_method == "target":
                    # Target encoding implementation
                    smoothing = data.get("target_encoding_smoothing", 1.0)
                    for col in categorical_cols:
                        df[col] = df[col].fillna('Unknown')
                        # Calculate target mean for each category
                        target_mean = df.groupby(col)[target].mean()
                        global_mean = df[target].mean()
                        
                        # Apply smoothing
                        counts = df.groupby(col).size()
                        smoothed_means = (counts * target_mean + smoothing * global_mean) / (counts + smoothing)
                        
                        # Map values
                        df[col] = df[col].map(smoothed_means).fillna(global_mean)
                    applied_steps.append("Target encoding applied to categorical features")
                    
                elif encoding_method == "binary":
                    # Simple binary encoding implementation
                    for col in categorical_cols:
                        df[col] = df[col].fillna('Unknown')
                        unique_vals = df[col].unique()
                        
                        # Create binary columns
                        n_bits = int(np.ceil(np.log2(len(unique_vals)))) if len(unique_vals) > 1 else 1
                        
                        for i in range(n_bits):
                            df[f"{col}_binary_{i}"] = 0
                        
                        # Encode each unique value
                        for idx, val in enumerate(unique_vals):
                            binary_rep = format(idx, f'0{n_bits}b')
                            for bit_idx, bit in enumerate(binary_rep):
                                df.loc[df[col] == val, f"{col}_binary_{bit_idx}"] = int(bit)
                        
                        # Drop original column
                        df = df.drop(columns=[col])
                    applied_steps.append("Binary encoding applied to categorical features")
                    
            except Exception as e:
                print(f"Encoding failed: {e}")
                # Fallback to label encoding
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = df[col].fillna('Unknown')
                    df[col] = le.fit_transform(df[col].astype(str))
                applied_steps.append(f"Encoding failed, applied label encoding as fallback: {str(e)}")

        # Enhanced target encoding with better error handling
        target_encoded = False
        if df[target].dtype == 'object':
            try:
                le_target = LabelEncoder()
                # Handle missing values in target
                df[target] = df[target].fillna('Unknown')
                df[target] = le_target.fit_transform(df[target].astype(str))
                target_encoded = True
                applied_steps.append("Target variable encoded")
            except Exception as e:
                print(f"Target encoding failed: {e}")
                applied_steps.append(f"Target encoding failed: {str(e)}")
                return jsonify(error=f"Target encoding failed: {str(e)}"), 500


        # 5. FEATURE SELECTION
        features = df.drop(columns=[target]).columns.tolist()
        
        # Remove constant features
        if data.get("remove_constant", False):
            constant_features = [col for col in features if df[col].nunique() <= 1]
            if constant_features:
                df = df.drop(columns=constant_features)
                features = [f for f in features if f not in constant_features]
                applied_steps.append(f"Removed {len(constant_features)} constant features")

        # Remove low variance features
        if data.get("remove_low_variance", False):
            variance_threshold = data.get("variance_threshold", 0.01)
            selector = VarianceThreshold(threshold=variance_threshold)
            numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_features) > 0:
                selected_features = selector.fit_transform(df[numeric_features])
                selected_feature_names = np.array(numeric_features)[selector.get_support()].tolist()
                removed_features = [f for f in numeric_features if f not in selected_feature_names]
                if removed_features:
                    df = df.drop(columns=removed_features)
                    features = [f for f in features if f not in removed_features]
                    applied_steps.append(f"Removed {len(removed_features)} low variance features")

        # Remove multicollinear features
        if data.get("remove_multicollinear", False):
            corr_threshold = data.get("multicollinear_threshold", 0.95)
            numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_features) > 1:
                corr_matrix = df[numeric_features].corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
                if high_corr_features:
                    df = df.drop(columns=high_corr_features)
                    features = [f for f in features if f not in high_corr_features]
                    applied_steps.append(f"Removed {len(high_corr_features)} highly correlated features")

        # Remove features with low correlation to target
        if data.get("remove_low_corr", False):
            low_corr_threshold = data.get("low_corr_threshold", 0.1)
            numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_features) > 0 and df[target].dtype in ['int64', 'float64']:
                correlations = df[numeric_features + [target]].corr()[target].abs()
                low_corr_features = correlations[correlations < low_corr_threshold].index.tolist()
                low_corr_features = [f for f in low_corr_features if f != target]
                if low_corr_features:
                    df = df.drop(columns=low_corr_features)
                    features = [f for f in features if f not in low_corr_features]
                    applied_steps.append(f"Removed {len(low_corr_features)} features with low correlation to target")

        # Select K best features - AUTO-DETECT VERSION
        if data.get("select_k_best", False):
            k_features = data.get("k_best_features", 10)
            
            # Only proceed if we have features to select from
            if len(features) > 0:
                # Adjust k_features to not exceed available features
                k_features = min(k_features, len(features))
                
                try:
                    # Auto-detect problem type based on target
                    def detect_problem_type(target_series):
                        """Auto-detect if problem is classification or regression"""
                        if target_series.dtype == 'object':
                            return 'classification'
                        elif target_series.nunique() <= 20 and target_series.dtype in ['int64', 'int32']:
                            return 'classification'
                        else:
                            return 'regression'
                    
                    problem_type = detect_problem_type(df[target])
                    
                    if problem_type == 'classification':
                        # Use f_classif for classification
                        selector = SelectKBest(score_func=f_classif, k=k_features)
                        X_selected = selector.fit_transform(df[features], df[target])
                        selected_features = np.array(features)[selector.get_support()].tolist()
                        score_func_used = 'f_classif'
                    else:
                        # Use f_regression for regression
                        selector = SelectKBest(score_func=f_regression, k=k_features)
                        X_selected = selector.fit_transform(df[features], df[target])
                        selected_features = np.array(features)[selector.get_support()].tolist()
                        score_func_used = 'f_regression'
                    
                    # Actually remove the unselected features from the DataFrame
                    features_to_remove = [f for f in features if f not in selected_features]
                    if features_to_remove:
                        df = df.drop(columns=features_to_remove)
                        features = selected_features  # Update features list
                        applied_steps.append(f"Selected {len(selected_features)} best features using {score_func_used} (auto-detected {problem_type})")
                        
                except Exception as e:
                    print(f"Feature selection failed: {e}")
                    applied_steps.append(f"Feature selection failed: {str(e)}")

        # Recursive Feature Elimination
        if data.get("enable_rfe", False):
            rfe_n_features = data.get("rfe_features", 10)
            rfe_estimator = data.get("rfe_estimator", "linear")
            rfe_step = data.get("rfe_step", 1)
            
            if len(features) > 0:
                rfe_n_features = min(rfe_n_features, len(features))
                
                try:
                    # Auto-detect problem type
                    problem_type = detect_problem_type(df[target])
                    
                    # Select appropriate estimator
                    if rfe_estimator == "linear":
                        if problem_type == 'classification':
                            estimator = LogisticRegression(max_iter=1000)
                        else:
                            estimator = LinearRegression()
                    elif rfe_estimator == "tree":
                        if problem_type == 'classification':
                            estimator = DecisionTreeClassifier(random_state=42)
                        else:
                            estimator = DecisionTreeRegressor(random_state=42)
                    elif rfe_estimator == "forest":
                        if problem_type == 'classification':
                            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
                        else:
                            estimator = RandomForestRegressor(n_estimators=10, random_state=42)
                    elif rfe_estimator == "svm":
                        if problem_type == 'classification':
                            estimator = SVC(kernel='linear')
                        else:
                            estimator = SVR(kernel='linear')
                    
                    # Apply RFE
                    rfe = RFE(estimator=estimator, n_features_to_select=rfe_n_features, step=rfe_step)
                    X_rfe = rfe.fit_transform(df[features], df[target])
                    selected_features = np.array(features)[rfe.support_].tolist()
                    
                    # Remove unselected features from DataFrame
                    features_to_remove = [f for f in features if f not in selected_features]
                    if features_to_remove:
                        df = df.drop(columns=features_to_remove)
                        features = selected_features
                        applied_steps.append(f"RFE selected {len(selected_features)} features using {rfe_estimator} estimator")
                        
                except Exception as e:
                    print(f"RFE failed: {e}")
                    applied_steps.append(f"RFE failed: {str(e)}")
            


        # 6. FEATURE SCALING
        scaling_method = data.get("scaling_method", "none")
        numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        
        if scaling_method != "none" and len(numeric_features) > 0:
            if scaling_method == "standard":
                scaler = StandardScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append("Standard scaling applied")
                
            elif scaling_method == "minmax":
                min_val = data.get("minmax_min", 0)
                max_val = data.get("minmax_max", 1)
                scaler = MinMaxScaler(feature_range=(min_val, max_val))
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append(f"Min-Max scaling applied (range: {min_val} to {max_val})")
                
            elif scaling_method == "robust":
                scaler = RobustScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append("Robust scaling applied")
                
            elif scaling_method == "maxabs":
                scaler = MaxAbsScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append("Max absolute scaling applied")
                
            elif scaling_method in ["quantile_uniform", "quantile_normal"]:
                output_dist = "uniform" if scaling_method == "quantile_uniform" else "normal"
                scaler = QuantileTransformer(output_distribution=output_dist)
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append(f"Quantile transformation applied ({output_dist} distribution)")
                
            elif scaling_method in ["power_yeo", "power_box"]:
                method = "yeo-johnson" if scaling_method == "power_yeo" else "box-cox"
                try:
                    scaler = PowerTransformer(method=method)
                    df[numeric_features] = scaler.fit_transform(df[numeric_features])
                    applied_steps.append(f"Power transformation applied ({method})")
                except Exception as e:
                    print(f"Power transformation failed: {e}")
                    
            elif scaling_method == "normalizer":
                norm = data.get("normalizer_norm", "l2")
                scaler = Normalizer(norm=norm)
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                applied_steps.append(f"Normalization applied ({norm} norm)")

        # Check if dataset is too small after preprocessing
        if df.shape[0] < 10:
            return jsonify(error="Dataset too small after preprocessing. Please use less aggressive settings."), 400

        # Save processed file
        new_filename = f"processed_{filename}"
        new_path = os.path.join(UPLOAD_FOLDER, new_filename)
        df.to_csv(new_path, index=False)

        return jsonify({
            "filename": new_filename,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "remaining_features": df.columns.tolist(),
            "applied_steps": applied_steps,
            "target_encoded": target_encoded
        })

    except Exception as e:
        print("Enhanced preprocessing error:", str(e))
        return jsonify(error=f"Preprocessing failed: {str(e)}"), 500





MODEL_FACTORY = {
    # Regression
    "LinearRegression"  : lambda p: LinearRegression(**p),
    "Ridge"             : lambda p: Ridge(**p),
    "Lasso"             : lambda p: Lasso(**p),
    "ElasticNet"        : lambda p: ElasticNet(**p),
    "DecisionTreeReg"   : lambda p: DecisionTreeRegressor(**p),
    "RandomForestReg"   : lambda p: RandomForestRegressor(**p),
    "GradBoostReg"      : lambda p: GradientBoostingRegressor(**p),
    "KNNReg"            : lambda p: KNeighborsRegressor(**p),
    "SVMReg"            : lambda p: SVR(**p),
    "XGBReg"            : lambda p: XGBRegressor(**p) if XGBRegressor else (_ for _ in ()).throw(Exception("XGBoost not installed")),
    "MLPReg"            : lambda p: MLPRegressor(hidden_layer_sizes=(int(p.get("hidden_layer_sizes",100)),)),
    # Classification
    "LogisticRegression": lambda p: LogisticRegression(max_iter=1000,**p),
    "NaiveBayes"        : lambda p: GaussianNB(**p),
    "DecisionTree"      : lambda p: DecisionTreeClassifier(**p),
    "RandomForest"      : lambda p: RandomForestClassifier(**p),
    "GradBoost"         : lambda p: GradientBoostingClassifier(**p),
    "KNN"               : lambda p: KNeighborsClassifier(**p),
    "SVM"               : lambda p: SVC(probability=True, **p),
    "XGB"               : lambda p: XGBClassifier(**p) if XGBClassifier else (_ for _ in ()).throw(Exception("XGBoost not installed")),
    "MLP"               : lambda p: MLPClassifier(hidden_layer_sizes=(int(p.get("hidden_layer_sizes",100)),), max_iter=300),
}
 

@app.route('/train', methods=['POST'])
def train_model():
    """
    Enhanced training endpoint with comprehensive error handling and validation
    """
    try:
        payload = request.get_json()
        filename = payload.get("filename")
        target = payload.get("target")
        model_id = payload.get("model_id")
        params = payload.get("params", {})
        
        # Validate required parameters
        if not all([filename, target, model_id]):
            return jsonify(error="Missing required parameters: filename, target, or model_id"), 400
        
        # Validate file exists
        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify(error="Dataset file not found"), 404
        
        # Load and validate dataset
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return jsonify(error=f"Failed to load dataset: {str(e)}"), 400
        
        # Validate target column exists
        if target not in df.columns:
            return jsonify(error=f"Target column '{target}' not found in dataset"), 400
        
        # Check dataset size
        if df.shape[0] < 10:
            return jsonify(error="Dataset too small for training (minimum 10 rows required)"), 400
        
        # Prepare features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Validate we have features
        if X.shape[1] == 0:
            return jsonify(error="No features available for training"), 400
        
        # Ensure all data is numeric before training
        for col in X.select_dtypes(include=['object']).columns:
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            except Exception as e:
                return jsonify(error=f"Failed to encode feature '{col}': {str(e)}"), 400
        
        # Ensure target is numeric
        if y.dtype == 'object':
            try:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            except Exception as e:
                return jsonify(error=f"Failed to encode target variable: {str(e)}"), 400
        
        # Verify all data is numeric
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            return jsonify(error="Data contains non-numeric values after preprocessing"), 400
        
        # Auto-detect problem type
        def detect_problem_type(target_series):
            """Auto-detect if problem is classification or regression"""
            if target_series.dtype == 'object':
                return 'classification'
            elif target_series.nunique() <= 20 and target_series.dtype in ['int64', 'int32']:
                return 'classification'
            else:
                return 'regression'
        
        problem_type = detect_problem_type(y)
        
        # Validate model exists
        if model_id not in MODEL_FACTORY:
            available_models = list(MODEL_FACTORY.keys())
            return jsonify(error=f"Unknown model ID '{model_id}'. Available models: {available_models}"), 400
        
        # Clean and validate parameters
        cleaned_params = {}
        for key, value in params.items():
            if value is not None and value != "":
                try:
                    # Convert string numbers to appropriate types
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        cleaned_params[key] = float(value) if '.' in value else int(value)
                    else:
                        cleaned_params[key] = value
                except (ValueError, TypeError):
                    cleaned_params[key] = value
        
        # Initialize model with error handling
        try:
            model = MODEL_FACTORY[model_id](cleaned_params)
        except Exception as e:
            return jsonify(error=f"Model initialization failed: {str(e)}"), 500
        
        # Train/Test Split with stratification for classification
        try:
            if problem_type == 'classification' and len(np.unique(y)) > 1:
                # Use stratified split for classification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # Regular split for regression or single-class classification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception as e:
            return jsonify(error=f"Failed to split dataset: {str(e)}"), 400
        
        # Train the model
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            return jsonify(error=f"Model training failed: {str(e)}"), 500
        
        # Make predictions and calculate metrics
        try:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            if problem_type == 'regression':
                # Regression metrics
                train_score = r2_score(y_train, y_pred_train)
                test_score = r2_score(y_test, y_pred_test)
                
                metrics = {
                    "r2_score": float(test_score),
                    "mae": float(mean_absolute_error(y_test, y_pred_test)),
                    "mse": float(mean_squared_error(y_test, y_pred_test)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                    "train_score": float(train_score),
                    "test_score": float(test_score)
                }
            else:
                # Classification metrics
                train_score = accuracy_score(y_train, y_pred_train)
                test_score = accuracy_score(y_test, y_pred_test)
                
                metrics = {
                    "accuracy": float(test_score),
                    "precision": float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                    "train_score": float(train_score),
                    "test_score": float(test_score)
                }
                
        except Exception as e:
            return jsonify(error=f"Failed to calculate metrics: {str(e)}"), 500
        
        # Save model and metadata
        try:
            model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            
            # Save feature names for future inference
            features_path = os.path.join(MODEL_FOLDER, f"{model_id}_features.txt")
            with open(features_path, "w") as f:
                f.write(",".join(X.columns.tolist()))
            
            # Save model metadata
            metadata = {
                "model_id": model_id,
                "problem_type": problem_type,
                "target_column": target,
                "feature_count": X.shape[1],
                "training_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "parameters": cleaned_params,
                "metrics": metrics
            }
            
            metadata_path = os.path.join(MODEL_FOLDER, f"{model_id}_metadata.json")
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            return jsonify(error=f"Failed to save model: {str(e)}"), 500
        
        # Check for potential overfitting
        overfitting_warning = None
        if abs(train_score - test_score) > 0.2:
            overfitting_warning = f"Potential overfitting detected: Train score ({train_score:.3f}) significantly higher than test score ({test_score:.3f})"
        
        # Prepare response
        response = {
            "message": "Model trained successfully",
            "model_id": model_id,
            "problem_type": problem_type,
            "model_path": model_path,
            "metrics": metrics,
            "dataset_info": {
                "total_samples": df.shape[0],
                "features": X.shape[1],
                "training_samples": X_train.shape[0],
                "test_samples": X_test.shape[0]
            },
            "parameters_used": cleaned_params
        }
        
        if overfitting_warning:
            response["warning"] = overfitting_warning
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify(error=f"Training failed: {str(e)}"), 500


@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
        r2_score, mean_absolute_error, mean_squared_error
    )

    data = request.get_json()
    filename = data.get("filename")
    target = data.get("target")
    model_id = data.get("model_id")
    problem_type = data.get("problem_type", "regression")

    if not all([filename, target, model_id]):
        return jsonify(error="Missing parameters"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(model_path) or not os.path.exists(file_path):
        return jsonify(error="Model or dataset not found"), 404

    df = pd.read_csv(file_path)
    if target not in df.columns:
        return jsonify(error="Target column missing"), 400

    X = df.drop(columns=[target])
    y = df[target]
    model = joblib.load(model_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    plot_data = {}
    metrics = {}

    if problem_type == "regression":
        errors = y_test - y_pred
        metrics = {
            "R-squared": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "Train Score": model.score(X_train, y_train),
            "Test Score": model.score(X_test, y_test)
        }
        plot_data = {
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist(),
            "residuals": errors.tolist()
        }
    else:
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 score": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Train Score": model.score(X_train, y_train),
            "Test Score": model.score(X_test, y_test),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        plot_data = {
            "confusion": metrics["Confusion Matrix"]
        }

    return jsonify(
        metrics={k: v for k, v in metrics.items() if k != "Confusion Matrix"},
        plots=plot_data,
        train_score=metrics.get("Train Score"),
        test_score=metrics.get("Test Score")
    )


@app.route('/download_model', methods=['GET'])
def download_model():
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify(error="Missing model_id"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify(error="Model not found"), 404

    return send_file(model_path, as_attachment=True)

def build_streamlit_zip(model_path: str, feature_list: list[str]) -> io.BytesIO:
    """
    Creates an in‑memory .zip that contains:
      •   model.pkl
      •   app.py      (auto‑generated Streamlit UI)
      •   requirements.txt
    Returns: BytesIO ready to send with send_file().
    """
    # ‼️ basic Streamlit template (Jinja2)
    st_template = Template(textwrap.dedent("""
        import streamlit as st, pandas as pd, joblib

        model = joblib.load("model.pkl")
        st.set_page_config(page_title="ZeroML Prediction App", layout="centered")

        st.title("🔮 ZeroML Prediction App")
        st.write("Enter feature values and click **Predict**")

        # gather inputs
        inputs = {}
        {% for f in features %}
        inputs["{{ f }}"] = st.number_input("{{ f }}")
        {% endfor %}

        if st.button("Predict"):
            X = pd.DataFrame([inputs])
            pred = model.predict(X)[0]
            st.success(f"✅ Prediction: {pred}")
    """))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # 1) model
        z.write(model_path, arcname="model.pkl")

        # 2) app.py
        z.writestr("app.py", st_template.render(features=feature_list))

        # 3) requirements
        z.writestr("requirements.txt",
                   "streamlit\npandas\nscikit-learn\njoblib\n")

    buf.seek(0)
    return buf

@app.route("/generate_streamlit", methods=["POST"])
def generate_streamlit():
    """
    Body  → { "model_id": "RandomForestReg" }
    Returns a .zip file the browser will download.
    """
    data     = request.get_json(force=True)
    model_id = data.get("model_id")

    if not model_id:
        return jsonify(error="model_id missing"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify(error="model not found – train first"), 404

    # retrieve the feature list saved alongside the model during training
    meta_path = os.path.join(MODEL_FOLDER, f"{model_id}_features.txt")
    if not os.path.exists(meta_path):
        return jsonify(error="feature metadata not found"), 500
    with open(meta_path) as f:
        feature_list = f.read().strip().split(",")

    zip_buf = build_streamlit_zip(model_path, feature_list)

    return send_file(zip_buf,
                     mimetype="application/zip",
                     as_attachment=True,
                     download_name=f"{model_id}_streamlit_app.zip")


def build_api_zip(model_path: str, feature_list: list[str]) -> io.BytesIO:
    """
    Creates a zip file in memory containing:
      - serve_model.py (Flask API)
      - model.pkl
      - requirements.txt
    """
    flask_template = Template(textwrap.dedent("""
        from flask import Flask, request, jsonify
        import pandas as pd
        import joblib

        app = Flask(__name__)
        model = joblib.load("model.pkl")

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json(force=True).get("input", [])
            df = pd.DataFrame(data)
            pred = model.predict(df).tolist()
            return jsonify({"predictions": pred})

        if __name__ == '__main__':
            app.run(debug=True)
    """))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(model_path, arcname="model.pkl")

        z.writestr("serve_model.py", flask_template.render(features=feature_list))
        z.writestr("requirements.txt", "flask\npandas\nscikit-learn\njoblib\n")

    buf.seek(0)
    return buf
@app.route("/generate_api_script", methods=["POST"])
def generate_api_script():
    data     = request.get_json(force=True)
    model_id = data.get("model_id")

    if not model_id:
        return jsonify(error="model_id missing"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify(error="model not found"), 404

    meta_path = os.path.join(MODEL_FOLDER, f"{model_id}_features.txt")
    if not os.path.exists(meta_path):
        return jsonify(error="feature metadata not found"), 500
    with open(meta_path) as f:
        features = f.read().strip().split(",")

    zip_buf = build_api_zip(model_path, features)
    return send_file(zip_buf,
                     mimetype="application/zip",
                     as_attachment=True,
                     download_name=f"{model_id}_api_script.zip")


if __name__ == "__main__":
    app.run(debug=True)
