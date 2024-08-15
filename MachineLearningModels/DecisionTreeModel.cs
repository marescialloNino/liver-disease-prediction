
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using liver_disease_prediction.dataModels;
using liver_disease_prediction.utility;
using System.Collections.Generic;
using System.Linq;
using System;


namespace liver_disease_prediction.MachineLearningModels
{
    public class DecisionTreeModel : MachineLearningModel
    {
        
        public DecisionTree Tree { get; set; }
        public DecisionVariable[] Features { get; set; }

        /// <summary>
        /// Initializes a new instance of the DecisionTreeModel class with predefined features.
        /// </summary>
        public DecisionTreeModel()
        {
            Features = new DecisionVariable[]
                            {
                                new DecisionVariable("Age", DecisionVariableKind.Continuous),
                                new DecisionVariable("Gender", 2),
                                new DecisionVariable("Direct Bilirubin", DecisionVariableKind.Continuous),
                                new DecisionVariable("Alkaline Phosphatase", DecisionVariableKind.Continuous),
                                new DecisionVariable("Aspartate Aminotransferase", DecisionVariableKind.Continuous),
                                new DecisionVariable("Total Protiens", DecisionVariableKind.Continuous),
                                new DecisionVariable("Albumin and Globulin Ratio", DecisionVariableKind.Continuous)
                            };
            Tree = new DecisionTree(Features, 2);
        }


        /// <summary>
        /// Trains the Decision Tree model with specified hyperparameters.
        /// </summary>
        /// <param name="records">Training data as a list of LiverPatientRecord.</param>
        /// <param name="join">Parameter for C45 algorithm's minimum number of samples per leaf.</param>
        /// <param name="maxHeight">Parameter for C45 algorithm's maximum height of the tree.</param>
        public void Train(List<LiverPatientRecord> records, double join, double maxHeight)
        {

            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);

            // Create an instance of the C4.5 learning algorithm
            C45Learning teacher = new C45Learning(Features)
            {
                Join = (int)join,
                MaxHeight = (int)maxHeight
            };

            // Use the learning algorithm to induce the tree
            Tree = teacher.Learn(inputs, outputs);
        }

        /// <summary>
        /// Predicts outcomes for the given records using the trained Decision Tree model.
        /// </summary>
        /// <param name="records">List of LiverPatientRecord for which to predict outcomes.</param>
        /// <returns>An array of integer predictions where each element corresponds to a record.</returns>
        public override int[] Predict(List<LiverPatientRecord> records)
        {
            (double[][] inputs, _) = DataUtility.recordsToInputsOutputs(records);
            return Tree.Decide(inputs);
        }

        /// <summary>
        /// Performs k-fold cross-validation with hyperparameter tuning and evaluates model performance.
        /// It returns the best parameters found and their corresponding performance metrics.
        /// </summary>
        /// <param name="foldedTrainSet">List of data folds for cross-validation.</param>
        /// <param name="parameterRanges">Dictionary with ranges for 'Join' and 'MaxHeight' parameters.</param>
        /// <returns>Tuple containing the best join, best max height, and performance metrics (accuracy, precision, recall, f1).</returns>
        public (double bestJoin, double bestMaxHeight, double[] bestMetrics) CrossValidation(
            List<List<LiverPatientRecord>> foldedTrainSet,
            Dictionary<string, double[]> parameterRanges)
        {

            List<List<LiverPatientRecord>> tempTrainingSetsList = new List<List<LiverPatientRecord>>();
            List<List<LiverPatientRecord>> tempValidationSetsList = new List<List<LiverPatientRecord>>();


            for (int i = 0; i < foldedTrainSet.Count; i++)
            {

                // Get all lists except the one at index i
                List<LiverPatientRecord> tempTrainSet = foldedTrainSet
                                            .Where((_, index) => index != i) // Filter out the current index
                                            .SelectMany(x => x) // Flatten the lists into a single list
                                            .ToList();
                tempTrainingSetsList.Add(tempTrainSet);

                List<LiverPatientRecord> tempValidationSet = foldedTrainSet[i];
                tempValidationSetsList.Add(tempValidationSet);

            }

            double bestAccuracy = 0.0;
            double bestPrecision = 0.0;
            double bestRecall = 0.0;
            double bestF1 = 0.0;
            double bestJoin = 0.0;
            double bestMaxHeight = 0.0;

            foreach (double join in parameterRanges["Join"])
            {
                foreach (double maxHeight in parameterRanges["MaxHeight"])
                {
                    List<double> accuracies = new List<double>();
                    List<double> precisions = new List<double>();
                    List<double> recalls = new List<double>();
                    List<double> f1Scores = new List<double>();

                    for (int j = 0; j < tempTrainingSetsList.Count; j++)
                    {
                        Train(tempTrainingSetsList[j], join, maxHeight);

                        int[] validationSetPredictions = Predict(tempValidationSetsList[j]);
                        (_, int[] validationSetOutputs) = DataUtility.recordsToInputsOutputs(tempValidationSetsList[j]);


                        (double accuracy,
                        double precision,
                        double recall,
                        double f1score) = MachineLearningModel.ComputeMetrics(tempValidationSetsList[j], validationSetPredictions);
                        accuracies.Add(accuracy);
                        precisions.Add(precision);
                        recalls.Add(recall);
                        f1Scores.Add(f1score);
                    }

                    double averageAccuracy = accuracies.Average();
                    double averagePrecision = precisions.Average();
                    double averageRecall = recalls.Average();
                    double averageF1score = f1Scores.Average();


                    if (averageF1score > bestF1)
                    {
                        bestAccuracy = averageAccuracy;
                        bestPrecision = averagePrecision;
                        bestRecall = averageRecall;
                        bestF1 = averageF1score;
                        bestJoin = join;
                        bestMaxHeight = maxHeight;
                    }

                }


            }
            double[] bestMetrics = new double[] { bestAccuracy, bestPrecision, bestRecall, bestF1 };

            Console.WriteLine("\nBest parameters after cross validation are:\n");
            Console.WriteLine($"Join: {bestJoin}");
            Console.WriteLine($"MaxHeight: {bestMaxHeight}");
            Console.WriteLine("\nTraining set Metrics for best parameters:\n");
            Console.WriteLine($"Accuracy: {bestAccuracy} , Precision: {bestPrecision}");
            Console.WriteLine($"Recall: {bestRecall}, F1 score: {bestF1}");


            return (bestJoin, bestMaxHeight, bestMetrics);
        }


    }
}
