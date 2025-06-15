from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
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


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "✅ Backend is running! Use the API endpoints like /upload, /train, etc."

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

    df = pd.read_csv(file_path)
    preview = df.head(5).to_dict(orient='records')
    return jsonify({
        'filename': file.filename,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'preview': preview
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

    return jsonify(analysis)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        filename = data.get("filename")
        target = data.get("target")

        if not filename or not target:
            return jsonify(error="filename or target missing"), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify(error="file not found"), 404

        df = pd.read_csv(path)

        # 1. Drop missing values
        if data.get("dropNA"):
            df = df.dropna()

        # 2. Remove duplicates
        if data.get("removeDuplicates"):
            df = df.drop_duplicates()

        # 3. Encode categorical features
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception as e:
                print(f"Label encoding failed on column: {col}", e)

        # 4. Remove outliers using IQR
        if data.get("removeOutliers"):
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        # 5. Drop features with low correlation to target
        if data.get("dropLowCorr"):
            if target in df.columns:
                corrs = df.corr()[target].abs()
                keep = corrs[corrs > 0.1].index.tolist()  # 0.1 = correlation threshold
                df = df[keep]

        # 6. Normalize (Min-Max Scaling)
        if data.get("normalize"):
            features = df.drop(columns=[target], errors='ignore').columns
            scaler = MinMaxScaler()
            df[features] = scaler.fit_transform(df[features])

        # 7. Standardize (Z-Score Scaling)
        if data.get("standardize"):
            features = df.drop(columns=[target], errors='ignore').columns
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])

        # Save processed file
        new_filename = f"processed_{filename}"
        df.to_csv(os.path.join(UPLOAD_FOLDER, new_filename), index=False)

        return jsonify({
            "filename": new_filename,
            "shape": df.shape,
            "columns": df.columns.tolist()
        })

    except Exception as e:
        print("Preprocessing Error:", e)
        return jsonify(error=str(e)), 500




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
    payload = request.get_json()

    filename = payload.get("filename")
    target   = payload.get("target")
    model_id = payload.get("model_id")
    params   = payload.get("params", {})

    if not all([filename, target, model_id]):
        return jsonify(error="Missing parameters"), 400

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify(error="file not found"), 404

    df = pd.read_csv(path)
    if target not in df.columns:
        return jsonify(error="Target column not found in dataset"), 400

    X = df.drop(columns=[target])
    y = df[target]

    # Choose model
    if model_id not in MODEL_FACTORY:
        return jsonify(error="Unknown model ID"), 400

    try:
        model = MODEL_FACTORY[model_id](params)
    except Exception as e:
        return jsonify(error=f"Model init failed: {str(e)}"), 500

    # Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Determine problem type and compute appropriate metric
        if y.dtype.kind in "ifu":  # regression
            preds = model.predict(X_test)
            metrics = {"r2": float(r2_score(y_test, preds))}
        else:  # classification
            preds = model.predict(X_test)
            metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    except Exception as e:
        return jsonify(error=f"Training failed: {str(e)}"), 500

    # Save model
    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    joblib.dump(model, model_path)

    return jsonify({
        "message": "Model trained successfully",
        "model_path": model_path,
        "metrics": metrics
    })

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    data = request.get_json()
    filename  = data.get("filename")
    target    = data.get("target")
    model_id  = data.get("model_id")
    problem_type = data.get("problem_type", "regression")  # fallback if not provided


    if not all([filename, target, model_id]):
        return jsonify(error="Missing parameters"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    file_path  = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(model_path):
        return jsonify(error="Model not found. Train first."), 404
    if not os.path.exists(file_path):
        return jsonify(error="Data file not found"), 404

    df = pd.read_csv(file_path)
    if target not in df.columns:
        return jsonify(error="Target column missing"), 400

    X = df.drop(columns=[target])
    y = df[target]

    # Reload trained model
    model = joblib.load(model_path)

    # Use the same 80/20 split so metrics match training step
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test)

    if problem_type == "regression":
  # Regression metrics
        metrics = {
            "r2"   : float(r2_score(y_test, y_pred)),
            "MAE"  : float(mean_absolute_error(y_test, y_pred)),
            "MSE"  : float(mean_squared_error(y_test, y_pred)),
            "RMSE" : float(mean_squared_error(y_test, y_pred) ** 0.5),

        }
    else:  # Classification metrics
        metrics = {
            "accuracy" : float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall"   : float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1"       : float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    return jsonify(metrics=metrics)

from flask import send_file  # ensure this import is present at the top

@app.route('/download_model', methods=['GET'])
def download_model():
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify(error="Missing model_id"), 400

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify(error="Model not found"), 404

    return send_file(model_path, as_attachment=True)



if __name__ == "__main__":
    app.run(debug=True)
