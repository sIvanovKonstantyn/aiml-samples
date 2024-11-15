package unsupervised_learning;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.Remove;

public class WekaUnsupervisedLearningDimensionalityReductionExample {

    public static void main(String[] args) {
        try {
            // 1. Load initial dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("datasets/laptops.arff");
            Instances dataset = source.getDataSet();

            // 2. REMOVE target attribute - since now our data is unlabeled
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }

            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("" + (dataset.classIndex() + 1));
            removeFilter.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, removeFilter);


            // 3.Create new model or load the existing one (if present)
            PrincipalComponents pca = new PrincipalComponents();
            pca.setMaximumAttributes(2); //Data should be educed to 2 attributes
            pca.setInputFormat(dataset);

            // 4. Apply PCA to data
            Instances reducedDataset = Filter.useFilter(dataset, pca);

            // 5.Show results
            System.out.println("Initial # of attributes: " + dataset.numAttributes());
            System.out.println("# of attributes after PCA: " + reducedDataset.numAttributes());
            System.out.println("\nFinal data:");
            System.out.println(reducedDataset);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
