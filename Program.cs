using liver_disease_prediction.dataModels;
using liver_disease_prediction.MachineLearningModels;
using liver_disease_prediction.utility;

internal class Program
{
    private static void Main(string[] args)
    {
        List<LiverPatientRecord> records = DataUtility.LoadDataFromCsv("C:\\Users\\ukg11058\\source\\repos\\data-visualization\\indian_liver_patient.csv");
        (List<LiverPatientRecord>  trainSet, List<LiverPatientRecord> testSet) = DataUtility.SplitData(records);

        LogisticRegressionModel logRegModel = new LogisticRegressionModel();
        logRegModel.Train(trainSet);

        DecisionTreeModel treeModel = new DecisionTreeModel();
        treeModel.Train(trainSet);

        SVMModel svmModel = new SVMModel();
        svmModel.Train(trainSet); 


        int[] logRegPredictions = logRegModel.Predict(testSet);
        int[] treePredictions = treeModel.Predict(testSet);
        int[] svmPredictions = svmModel.Predict(testSet);

        double logRegAccuracy = logRegModel.Validate(testSet, logRegPredictions);
        Console.WriteLine($"Logistic Regression Accuracy: {logRegAccuracy}");

        double treeaccuracy = treeModel.Validate(testSet, treePredictions);
        Console.WriteLine($"Decision tree Accuracy: {treeaccuracy}");
        
        double svmAccuracy = svmModel.Validate(testSet, svmPredictions);
        Console.WriteLine($"SVM Accuracy: {svmAccuracy}");


    }
}
