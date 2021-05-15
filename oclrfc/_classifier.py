import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ._converter import RFC_to_OCL


class OCLRandomForestClassifier():
    def __init__(self, num_features: int, num_classes: int, opencl_filename = "temp.cl", max_depth: int = 2, num_trees: int = 10):
        self.num_features = num_features
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.num_trees = num_trees

        self.opencl_file = opencl_filename
        self.classifier = None

    def train(self, features, ground_truth):
        if len(features) != self.num_features:
            warnings.warn("Wrong number of features!")

        X, y = self._to_np(features, ground_truth)

        self.classifier = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.num_trees, random_state=0)

        self.classifier.fit(X, y)

        self.to_ocl_file(self.classifier, self.opencl_file)

    def to_ocl_file(self, classifier, filename):
        opencl_code = RFC_to_OCL(classifier)

        file1 = open(filename, "w")
        file1.write("// OpenCL RandomForestClassifier\n")
        file1.write("// num_classes = " + str(self.num_classes) + "\n")
        file1.write("// num_features = " + str(self.num_features) + "\n")
        file1.write("// max_depth = " + str(self.max_depth) + "\n")
        file1.write("// num_trees = " + str(self.num_trees) + "\n")
        file1.write(opencl_code)
        file1.close()

        self.opencl_file = filename

    def from_ocl_file(self, filename):
        self.opencl_file = filename

    def _to_np(self, features, ground_truth=None):

        feature_stack = np.asarray([np.asarray(f).ravel() for f in features]).T
        if ground_truth is None:
            return feature_stack, None
        else:
            # make the annotation 1-dimensional
            ground_truth_np = np.asarray(ground_truth).ravel()

            X = feature_stack
            y = ground_truth_np

            # remove all pixels from the feature and annotations which have not been annotated
            mask = y > 0
            X = X[mask]
            y = y[mask]

            return X, y

    def predict(self, features):
        if self.classifier is None:
            warning.warn("Classifier has not been trained")
            return None

        image = features[0]

        feature_stack, _ = self._to_np(features)

        result_1d = self.classifier.predict(feature_stack)  # we subtract 1 to make background = 0
        result_2d = result_1d.reshape(image.shape)

        return result_2d

    def predict_gpu(self, features):
        import pyclesperanto_prototype as cle

        output = cle.create_like(features[0])

        parameters = {}
        for i, f in enumerate(features):
            parameters['in' + str(i)] = f

        parameters['out'] = output

        cle.execute(None, self.opencl_file, "predict", features[0].shape, parameters)

        return output
