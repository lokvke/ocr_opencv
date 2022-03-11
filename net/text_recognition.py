import cv2


class TextRecognition:
    def __init__(self, model_file, alphabet_file):
        self.chars = self.load_alphabet(alphabet_file)
        self.model = self.load_model(model_file, self.chars)

    def __call__(self, image):
        outputs = self.infer(image)
        return outputs

    def load_model(self, model_file, chars):
        model = cv2.dnn_TextRecognitionModel(model_file)
        model.setDecodeType("CTC-greedy")
        model.setVocabulary(chars)

        # preprocess parameters
        scale = 1.0 / 127.5
        mean = (127.5, 127.5, 127.5)
        input_size = (100, 32)
        model.setInputParams(scale, input_size, mean)
        return model

    def load_alphabet(self, alphabet_file):
        with open(alphabet_file, 'r', encoding='utf-8') as fr:
            chars = fr.readlines()
        chars = [char[0] for char in chars]
        return chars

    def infer(self, image):
        outputs = self.model.recognize(image)
        return outputs
