using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using liver_disease_prediction.dataModels;
using liver_disease_prediction.MachineLearningModels;
using liver_disease_prediction.utility;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using liver_disease_prediction.utility;
using System.Collections.Generic;
using System.Linq;
using System;

internal class Program
{
    private static void Main(string[] args)
    {

        // ---------------------------------------DATA PREPARATION-----------------------------------------------------

        List<LiverPatientRecord> records = DataUtility.LoadDataFromCsv("indian_liver_patient.csv");
        (List<LiverPatientRecord> trainSet, List<LiverPatientRecord> testSet) = DataUtility.SplitData(records);

        List<List<LiverPatientRecord>> folds = DataUtility.SplitDataIntoFolds(trainSet);

        // ---------------------------------------LOGISTIC REGRESSION-----------------------------------------------------

        Console.WriteLine("\n---------LOGISTIC REGRESSION---------\n");

        LogisticRegressionModel logisticRegressionModel = new LogisticRegressionModel();


        Dictionary<string, double[]> logRegParameterRanges = new Dictionary<string, double[]>
        {
            { "Regularization", new double[] {1e-1, 1e-4, 1e-7} },
            { "Intercept", new double[] { 0.0, 1.0, 2.0 } }
        };

        (double bestRegularization, double bestIntercept, double[] logRegMetrics) = logisticRegressionModel.CrossValidation(folds, logRegParameterRanges);

        logisticRegressionModel.Train(trainSet, bestRegularization, bestIntercept);
        int[] logRegPredictions = logisticRegressionModel.Predict(testSet);
        


        (double accuracy, double precision, double recall, double f1Score) = MachineLearningModel.ComputeMetrics(testSet, logRegPredictions);
        

        Console.WriteLine("\nTest set Metrics for best parameters:\n");
        Console.WriteLine($"Accuracy: {accuracy} , Precision: {precision}");
        Console.WriteLine($"Recall: {recall}, F1 score: {f1Score}");


        // ---------------------------------------DECISION TREE-----------------------------------------------------

        Console.WriteLine("\n---------DECISION TREE---------\n");

        DecisionTreeModel treeModel = new DecisionTreeModel();

        Dictionary<string, double[]> treeParameterRanges = new Dictionary<string, double[]>
        {
            { "Join", new double[] { 1.0, 5.0, 10.0, 15.0} },
            { "MaxHeight", new double[] { 10.0, 20.0, 40.0 } }
        };

        (double bestJoin, double bestMaxHeight, double[] treeMetrics) = treeModel.CrossValidation(folds, treeParameterRanges);

        treeModel.Train(trainSet, bestJoin, bestMaxHeight);
        int[] treePredictions = treeModel.Predict(testSet);
        (double treeaccuracy, double treeprecision, double treerecall, double treef1Score) = MachineLearningModel.ComputeMetrics(testSet, treePredictions);

        Console.WriteLine("\nTest set Metrics for best parameters:\n");
        Console.WriteLine($"Accuracy: {treeaccuracy} , Precision: {treeprecision}");
        Console.WriteLine($"Recall: {treerecall}, F1 score: {treef1Score}");




        // ---------------------------------------SUPPORT VECTOR MACHINES-----------------------------------------------------


        Console.WriteLine("\n---------SUPPORT VECTOR MACHINES---------\n");

        SVMModel svm = new SVMModel();

        IKernel[] kernels = new IKernel[] { new Gaussian(), new Linear(), new ChiSquare()};

        double[] complexities = new double[] {1e-10, 1e-7, 1e-4};


        (IKernel bestKernel, double bestComplexity, double[] svmMetrics) = svm.CrossValidation(folds, kernels, complexities);
        

        svm.Train(trainSet, bestKernel, bestComplexity);
        int[] svmPredictions = svm.Predict(testSet);
        (double svmaccuracy, double svmprecision, double svmrecall, double svmf1Score) = MachineLearningModel.ComputeMetrics(testSet, svmPredictions);

        Console.WriteLine("\nTest set Metrics for best parameters:\n");
        Console.WriteLine($"Accuracy: {svmaccuracy} , Precision: {svmprecision}");
        Console.WriteLine($"Recall: {svmrecall}, F1 score: {svmf1Score}");


    }
}
