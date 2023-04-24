# Red-Wine-Quality-Prediction
This project uses machine learning to predict the quality of red wine based on various chemical properties. The dataset used in this project contains information about different red wine samples and their quality ratings on a scale of 0 to 10. The dataset can be found in the file "winequality-red.csv" in the "data" folder.

## Dependencies:

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

## Installation:

1.Clone the repository: git clone https://github.com/Sam-ops09/Red-Wine-Quality-Prediction.git
2.Navigate to the project directory: cd Red-Wine-Quality-Prediction
3.Install the required packages: pip install -r requirements.txt

## Usage:

1.Run the Jupyter notebook: jupyter notebook
2.Open the notebook "Red Wine Quality Prediction.ipynb" in your browser.
3.Follow the instructions in the notebook to train and evaluate the machine learning models.

Alternatively, you can run the script "predict_quality.py" to make predictions on a new dataset. The script takes a CSV file as input and outputs the predicted quality ratings for each sample in the file. To use the script, run the following command:
python predict_quality.py --input_file path/to/input_file.csv --output_file path/to/output_file.csv

## License:
This project is licensed under the MIT License. See the LICENSE file for details.

## References:
The dataset used in this project is from the UCI Machine Learning Repository. The original paper describing the dataset can be found at https://www.sciencedirect.com/science/article/pii/S0167923609001377.
