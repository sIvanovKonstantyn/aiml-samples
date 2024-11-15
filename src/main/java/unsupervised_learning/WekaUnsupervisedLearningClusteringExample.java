package unsupervised_learning;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.File;

public class WekaUnsupervisedLearningClusteringExample {
    private static final String LAPTOP_CLASSIFYING_MODEL_PATH = "models/SimpleKMeans_laptops.model";

    public static void main(String[] args) {
        try {
            // 1. Load initial dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("datasets/laptops.arff");
            Instances dataset = source.getDataSet();

            // 2. REMOVE target attribute - since now our data is unlabeled
            if (dataset.classIndex() != -1)
                dataset.setClassIndex(-1);

            // 3.Create new model or load the existing one (if present)
            SimpleKMeans loadedModel;
            if(new File(LAPTOP_CLASSIFYING_MODEL_PATH).exists()) {
                loadedModel = (SimpleKMeans) SerializationHelper.read(LAPTOP_CLASSIFYING_MODEL_PATH);
            } else {
                // Set nu,ber of clusters and seeds.
                loadedModel = new SimpleKMeans();
                loadedModel.setNumClusters(3);//This is equal to target amount f groups
                loadedModel.setSeed(1000); //This is we play with to achieve the best results
            }

            // 4. Train/ fine-tuning the model
            loadedModel.buildClusterer(dataset);
            System.out.println("\nModel was trained on the dataset!");

            // 5. Show clusterization results
            System.out.println("Cluster centroids coordinates:");
            Instances centroids = loadedModel.getClusterCentroids();
            for (int i = 0; i < centroids.numInstances(); i++) {
                System.out.println("Cluster " + i + ": " + centroids.instance(i));
            }

            // 6. Save model into the file
            SerializationHelper.write(LAPTOP_CLASSIFYING_MODEL_PATH, loadedModel);
            System.out.println("\nModel was saved!");


            ConverterUtils.DataSource newData = new ConverterUtils.DataSource("datasets/test_laptops.arff");
            Instances testDataset = newData.getDataSet();

            // 7. Matching with results:
            System.out.println("\nCluster matching:");
            for (int i = 0; i < testDataset.numInstances(); i++) {
                int cluster = loadedModel.clusterInstance(testDataset.instance(i));
                System.out.println("Instance " + i + " matches the cluster " + cluster);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
