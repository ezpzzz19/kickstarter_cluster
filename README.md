# Clustering Kickstart Project

This project analyzes Kickstarter data using clustering techniques.

## Setup Instructions

### 1. Virtual Environment
The project uses a Python virtual environment to manage dependencies.

**Activate the virtual environment:**
```bash
source venv/bin/activate
```

**Deactivate when done:**
```bash
deactivate
```

### 2. Dependencies
All required packages are listed in `requirements.txt` and have been installed in the virtual environment.

**To reinstall dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Running the Project
```bash
python main.py
```

## Project Structure
- `main.py` - Main analysis script
- `data/` - Contains the Kickstarter dataset
- `requirements.txt` - Python dependencies
- `venv/` - Virtual environment (do not commit to version control)

## Dependencies
- pandas - Data manipulation and analysis
- openpyxl - Excel file reading
- numpy - Numerical computing
- scikit-learn - Machine learning and clustering
- matplotlib - Plotting
- seaborn - Statistical data visualization
- jupyter - Interactive notebooks

## Python Interpreter
The project uses Python 3.12 with a virtual environment. The interpreter path is:
```
/Users/etiennetremblay/clustering_kickstart/venv/bin/python
```

To use this interpreter in your IDE:
1. Open your IDE settings
2. Set the Python interpreter to the path above
3. Make sure the virtual environment is activated when running scripts
# kickstarter_cluster
