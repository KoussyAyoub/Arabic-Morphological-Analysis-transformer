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

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

- The Nemlar Arabic corpus used in this project was provided by [source-name].
- The implementation of the sequence-to-sequence model was inspired by [source-name].

## Contact

For any questions or inquiries, please contact [your-email-address].

Feel free to explore the code and experiment with different hyperparameters to further improve the model's performance. Happy analyzing!
