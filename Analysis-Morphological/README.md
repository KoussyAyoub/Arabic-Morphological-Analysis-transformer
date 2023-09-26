# Arabic Morphological Analysis Transformer

This project implements a sequence-to-sequence model with a bidirectional encoder and beam search decoding for Arabic Morphological Analysis. The model is trained on the Nemlar Arabic corpus, which is included in the repository as a CSV file.

## Implementation Details

The main components of the implementation are as follows:

- **Data Preprocessing**: The `Extract.py` script is used to extract and preprocess the data from the Nemlar Arabic corpus. It generates a CSV file named `data.csv` in the `data` directory.

- **Model Architecture**: The core of the model is defined in the `Model` class in the `main.py` file. It consists of a bidirectional LSTM encoder and a decoder with attention mechanism. Positional embeddings are added to the encoder for better representation of word positions.

- **Training**: The `train_model.py` script is used to train the model. It initializes an instance of the `Model` class and performs the training process. The model is trained for a specified number of epochs using teacher forcing and the Adam optimizer.

- **Decoding**: The `decode` method in the `Model` class performs the decoding process using beam search decoding. It takes the encoder states, root batch, teacher forcing boolean, and epoch as inputs and returns a list of softmax outputs.

- **Evaluation**: The `evaluate_model.py` script is used to evaluate the trained model. It loads the trained model from the checkpoint file and evaluates it on a test word. The predicted root is displayed.

## Results

The trained model achieves high accuracy in extracting the optimal root from Arabic words. It demonstrates the effectiveness of the sequence-to-sequence model with bidirectional encoding and beam search decoding for Arabic Morphological Analysis.
This is a Flask web application that serves as a morphological analyzer. It takes an input word and predicts its morphological analysis, providing the root and other relevant information.

## Installation

1. Clone the repository to your local machine:
git clone https://github.com/your-username/your-repository.git
Copy


2. Install the required dependencies by running the following command:
pip install -r requirements.txt
Copy


3. Download the pre-trained model file `best_model.pt` and place it in the root directory of the project.

## Usage

1. Start the Flask application by running the following command:
python app.py
Copy


2. Open your web browser and go to `http://localhost:5000` to access the application.

3. Enter a word in the input field and click the "Analyser" button.

4. The predicted morphological analysis will be displayed in the output field.

5. To copy the output text, click the "Copy" button.

## File Descriptions

- `app.py`: This file contains the Flask application code, including the routes and HTML templates.

- `index.html`: This HTML file defines the structure and layout of the web application.

- `requirements.txt`: This file lists all the required Python packages and their versions.

- `best_model.pt`: This is the pre-trained model file used for predicting the morphological analysis.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.



## Acknowledgements

- The Nemlar Arabic corpus used in this project was provided by [(https://catalogue.elra.info/en-us/repository/browse/ELRA-W0042/)].
- The implementation of the sequence-to-sequence model was inspired by [(https://aclanthology.org/W19-4610.pdf)].

## Contact

For any questions or inquiries, please contact [ayoubkoussy0@gmail.com] or [aminetaifat528@gmail.com].

Feel free to explore the code and experiment with different hyperparameters to further improve the model's performance. Happy analyzing!
