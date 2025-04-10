from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import time
from rdkit import Chem
from rdkit.Chem import Descriptors

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Available property calculations
PROPERTY_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "NumRings": Descriptors.RingCount,
    "TPSA": Descriptors.TPSA,
    "MolLogP": Descriptors.MolLogP,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/calculate", methods=["POST"])
def calculate_properties():
    start_time = time.time()
    
    # Get file and properties
    file = request.files.get("file")
    properties = request.form.get("properties")
    
    if not file or not properties:
        return jsonify({"error": "Missing file or properties."}), 400
    
    properties = set(eval(properties))  # Convert string to list
    invalid_props = properties - PROPERTY_FUNCTIONS.keys()
    if invalid_props:
        return jsonify({"error": f"Invalid properties selected: {invalid_props}"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Read CSV and process
    df = pd.read_csv(file_path)
    if "SMILES" not in df.columns:
        return jsonify({"error": "CSV must contain a 'SMILES' column."}), 400
    
    results = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            # Extract Database_ID from the row or set as 'Unknown' if not available
            database_id = row.get("Database_ID", "Unknown")  # Get Database_ID, default to 'Unknown'
            result = {"Database_ID": database_id, "SMILES": row["SMILES"]}
            
            # Add properties with rounded values to two decimal places
            for prop in properties:
                value = PROPERTY_FUNCTIONS[prop](mol)
                if isinstance(value, float):
                    result[prop] = round(value, 2)  # Round to 2 decimals
                else:
                    result[prop] = value
            results.append(result)
        else:
            # Handle invalid SMILES string (e.g., if mol is None)
            results.append({"Database_ID": row.get("Database_ID", "Unknown"), "SMILES": row["SMILES"], **{prop: "NaN" for prop in properties}})
    
    result_df = pd.DataFrame(results)
    result_file = os.path.join(RESULTS_FOLDER, "results.csv")
    result_df.to_csv(result_file, index=False)
    
    return jsonify({
        "time_taken_seconds": round(time.time() - start_time, 2),
        "batch_files": ["results.csv"]
    })

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found."}), 404

if __name__ == "__main__":
    app.run(debug=True)
