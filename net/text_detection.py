import cv2


class TextDetection:
    def __init__(self, model_file, bin_thresh=0.3, poly_thresh=0.5, max_candidates=200, unclip_ratio=2.0):
        '''
        :param model_file: 模型文件
        :param bin_thresh: DBPostProcess中分割图进行二值化的阈值，默认值为0.3
        :param poly_thresh: DBPostProcess中对输出框进行过滤的阈值，低于此阈值的框不会输出
        :param max_candidates: DBPostProcess中输出的最大文本框数量，默认200
        :param unclip_ratio: DBPostProcess中对文本框进行放大的比例
        '''
        self.model = self.load_model(model_file, bin_thresh, poly_thresh, max_candidates, unclip_ratio)

    def __call__(self, image):
        outputs = self.infer(image)
        return outputs

    def load_model(self, model_file, bin_thresh, poly_thresh, max_candidates, unclip_ratio):
        model = cv2.dnn_TextDetectionModel_DB(model_file)
        model.setBinaryThreshold(bin_thresh)
        model.setPolygonThreshold(poly_thresh)
        model.setMaxCandidates(max_candidates)
        model.setUnclipRatio(unclip_ratio)

        # preprocess parameters
        scale = 1.0 / 255.0
        mean = (122.67891434, 116.66876762, 104.00698793)
        input_size = (736, 736)
        model.setInputParams(scale, input_size, mean)
        return model

    def infer(self, image):
        outputs = self.model.detect(image)
        return outputs
