import subprocess
import os

def main():
    # Step 1: Preprocess the data
    print("Starting data preprocessing...")
    subprocess.run(["python", "preprocessing.py", "raw/DM_project_24.csv"])
    print("Data preprocessing completed.\n")
    
    # Step 2: Train models and save the best one
    print("Starting model training and selection...")
    subprocess.run(["python", "training.py"])
    print("Model training and selection completed.\n")
    
    # Step 3: Apply the best model to test data and save results
    print("Starting model application...")
    subprocess.run(["python", "prediction.py"])
    print("Model application completed and results saved.")

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("processed", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    
    main()
