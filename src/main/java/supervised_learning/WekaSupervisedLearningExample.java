package supervised_learning;

import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.io.File;
import java.util.Random;

public class WekaSupervisedLearningExample {

    private static final String LAPTOP_CLASSIFYING_MODEL_PATH = "models/J48_laptops.model";

    public static void main(String[] args) {
        try {
            // 1. Load initial dataset
            DataSource source = new DataSource("datasets/laptops.arff");
            Instances dataset = source.getDataSet();

            // 2. Set target attribute (laptop name)
            if (dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes() - 1);

            // 3.Create new model or load the existing one (if present)
            Classifier loadedModel;
            if(new File(LAPTOP_CLASSIFYING_MODEL_PATH).exists()) {
                loadedModel = (Classifier) SerializationHelper.read(LAPTOP_CLASSIFYING_MODEL_PATH);
            } else {
                loadedModel = new J48();
            }

            // 4. Cross-validation model check
            // Как работает k-кратная кросс-валидация:
            // Разделение данных: Датасет делится на k равных частей (например, при 10-кратной кросс-валидации — на 10 частей).
            // Циклы обучения и тестирования: Каждый раз k-1 часть данных используется для обучения модели, а оставшаяся часть — для тестирования. Этот процесс повторяется k раз, так что каждая часть данных побывает в роли тестовой ровно один раз.
            // Сравнение и усреднение: После выполнения всех k итераций полученные значения метрик (например, точность, F-мера) усредняются, чтобы получить общую оценку модели.
            Evaluation evaluation = new Evaluation(dataset);
            evaluation.crossValidateModel(loadedModel, dataset, 10, new Random(1));

            // 5. Show model evaluation results
            System.out.println(evaluation.toSummaryString("\nClassification results:\n", false));
            System.out.println("Accuracy: " + evaluation.pctCorrect() + "%");
            System.out.println("Errors matrix:\n" + evaluation.toMatrixString());

            // 6. Train/ fine-tuning the model
            loadedModel.buildClassifier(dataset);
            System.out.println("\nModel was trained on the dataset!");

            // 7. Save model into the file
            SerializationHelper.write(LAPTOP_CLASSIFYING_MODEL_PATH, loadedModel);
            System.out.println("\nModel was saved!");


            DataSource newData = new DataSource("datasets/test_laptops.arff");
            Instances testDataset = newData.getDataSet();
            if (testDataset.classIndex() == -1)
                testDataset.setClassIndex(testDataset.numAttributes() - 1);

            // 8. Test the model on a new dataset
            System.out.println("\nClassification of the new data:");
            for (int i = 0; i < testDataset.numInstances(); i++) {
                double classLabel = loadedModel.classifyInstance(testDataset.instance(i));
                String predictedClass = testDataset.classAttribute().value((int) classLabel);
                System.out.println("Instance " + (i + 1) + ": " + predictedClass);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
