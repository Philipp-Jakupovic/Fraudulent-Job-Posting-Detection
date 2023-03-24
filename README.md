# Fraudulent-Job-Posting-Detection

This project uses BERT, PyTorch, and the Hugging Face Transformers library to develop a machine learning model for fraud detection in job postings using natural language processing. The model is trained on a dataset of labeled job postings, detecting fraudulent vs legitimate postings.

## Kaggle Competition Dataset

- Name: [Real or Fake] : Fake Job Description Prediction (https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Description: This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.
- Creator: The University of the Aegean | Laboratory of Information & Communication Systems Security
- License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your_username/fraudulent-job-posting-detection.git
```

2. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate

3. Install the required dependencies:
pip install -r requirements.txt

## Usage

1. Train and test the model using default settings:
python main.py

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
