from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

def preprocess(df):
    df = df.copy()
    df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
    df = df[df['Experience'] >= 0]

    for col in ['Income', 'CCAvg', 'Mortgage']:
        df[col] = df[col].apply(lambda x: max(x, 0))
        df[col] = np.log1p(df[col])

    df['HasMortgage'] = (df['Mortgage'] > 0).astype(int)
    df.drop(columns=['Experience', 'Mortgage'], inplace=True, errors='ignore')
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/success", methods=["GET", "POST"])
def success():
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(url_for("index"))

        df = pd.read_csv(file)
        df_input = preprocess(df)
        X = df_input[model.feature_names_in_]

        df["Prediction"] = model.predict(X)
        df["Probability"] = model.predict_proba(X)[:, 1] 

        return render_template("data.html", Y=df.to_html(classes='table', index=False))

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)























#---------------Alternate method to save csv file --------------------------------


# from flask import Flask, render_template, request, redirect, url_for
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# TEMP_FILE = "temp_results.csv"

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# model = joblib.load("model.pkl")


# def preprocess(df):
#     """Apply same preprocessing steps used during model training."""
#     df = df.copy()
#     df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
#     df = df[df['Experience'] >= 0]

#     for col in ['Income', 'CCAvg', 'Mortgage']:
#         df[col] = df[col].apply(lambda x: max(x, 0)) 
#         df[col] = np.log1p(df[col])

#     df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)
#     df.drop(columns=['Experience', 'Mortgage'], inplace=True, errors='ignore')

#     return df


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/success", methods=["GET", "POST"])
# def success():
#     if request.method == "POST":
#         file = request.files.get('file')
#         if not file or file.filename == '':
#             return redirect(url_for("index"))

#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
#         file.save(filepath)

#         df = pd.read_csv(filepath)
#         df_processed = preprocess(df)

#         X = df_processed[model.feature_names_in_]
#         df_processed.insert(0, 'Prediction', model.predict(X))
#         df_processed.to_csv(TEMP_FILE, index=False)

#         return redirect(url_for("success"))

#     if not os.path.exists(TEMP_FILE):
#         return redirect(url_for("index"))

#     df = pd.read_csv(TEMP_FILE)
#     return render_template("data.html", Y=df.to_html(classes='table', index=False))


# if __name__ == "__main__":
#     app.run(debug=True)

