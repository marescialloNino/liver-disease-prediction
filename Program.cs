using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression;
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

        /*
        int[] logRegPredictions = logRegModel.Predict(testSet);
        int[] treePredictions = treeModel.Predict(testSet);
        int[] svmPredictions = svmModel.Predict(testSet);

        double logRegAccuracy = logRegModel.Validate(testSet, logRegPredictions);
        Console.WriteLine($"Logistic Regression Accuracy: {logRegAccuracy}");

        double treeaccuracy = treeModel.Validate(testSet, treePredictions);
        Console.WriteLine($"Decision tree Accuracy: {treeaccuracy}");
        
        double svmAccuracy = svmModel.Validate(testSet, svmPredictions);
        Console.WriteLine($"SVM Accuracy: {svmAccuracy}");
        */


        CrossValidationResult<LogisticRegression, double[], int> logCrossValResult = logRegModel.CrossValidation(trainSet, 5);


        // Finally, access the measured performance.
        double logTrainingErrors = logCrossValResult.Training.Mean; // should be 0.30606060606060609 (+/- var. 0.083498622589531682)
        double logValidationErrors = logCrossValResult.Validation.Mean; // should be 0.3666666666666667 (+/- var. 0.023333333333333334)

        // If desired, compute an aggregate confusion matrix for the validation sets:
        GeneralConfusionMatrix logGcm = logRegModel.GenerateConfusionMatrix(trainSet, logCrossValResult);
        double logAccuracy = logGcm.Accuracy;
        double logError = logGcm.Error;

        Console.WriteLine("DECISION TREE CROSS VALIDATION RESULTS:");
        Console.WriteLine($"training errors: {logTrainingErrors}");
        Console.WriteLine($"validation errors: {logValidationErrors}");
        Console.WriteLine($"accuracy: {logAccuracy}");
        Console.WriteLine($"error: {logError}");






        CrossValidationResult<DecisionTree, double[], int> crossValResult =  treeModel.CrossValidation(trainSet, 5);


        // Finally, access the measured performance.
        double trainingErrors = crossValResult.Training.Mean; // should be 0.30606060606060609 (+/- var. 0.083498622589531682)
        double validationErrors = crossValResult.Validation.Mean; // should be 0.3666666666666667 (+/- var. 0.023333333333333334)

        // If desired, compute an aggregate confusion matrix for the validation sets:
        GeneralConfusionMatrix gcm = treeModel.GenerateConfusionMatrix(trainSet, crossValResult);
        double accuracy = gcm.Accuracy; 
        double error = gcm.Error;

        Console.WriteLine("DECISION TREE CROSS VALIDATION RESULTS:");
        Console.WriteLine($"training errors: {trainingErrors}");
        Console.WriteLine($"validation errors: {validationErrors}");
        Console.WriteLine($"accuracy: {accuracy}");
        Console.WriteLine($"error: {error}");

    }
}
