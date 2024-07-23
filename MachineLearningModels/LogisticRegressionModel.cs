
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning;
using Accord.Statistics.Analysis;
using liver_disease_prediction.utility;
using System.Collections.Generic;


namespace liver_disease_prediction.MachineLearningModels
{
    public class LogisticRegressionModel:MachineLearningModel
    {
        private LogisticRegression LogRegModel { get; set; }


        /// <summary>
        /// Constructor that initializes a new instance of the LogisticRegression model.
        /// </summary>
        public LogisticRegressionModel()
        {
            LogRegModel = new LogisticRegression() { };
        }


        /// <summary>
        /// Trains the logistic regression model using a list of liver patient records.
        /// </summary>
        /// <param name="records">List of liver patient records to train the model.</param>
        public void Train(List<LiverPatientRecord> records, double regularization, double intercept)
        {
            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);

            LogRegModel.Intercept = intercept;

            IterativeReweightedLeastSquares<LogisticRegression> teacher = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                Regularization = regularization,
                MaxIterations = 100
            };


            LogRegModel = teacher.Learn(inputs, outputs);

        }


        /// <summary>
        /// Predicts the outcomes for a given list of liver patient records.
        /// </summary>
        /// <param name="records">The records for which predictions are to be made.</param>
        /// <returns>An array of predictions where 1 indicates presence of disease and 0 indicates absence.</returns>
        public override int[] Predict(List<LiverPatientRecord> records)
        {
            (double[][] inputs, _) = DataUtility.recordsToInputsOutputs(records);
            bool[] predictions = LogRegModel.Decide(inputs);
            int[] binaryPredictions = Array.ConvertAll(predictions, p => p ? 1 : 0);
            return binaryPredictions;

        }


        /// Performs k-fold cross-validation with hyperparameter tuning and evaluates model performance using a confusion matrix.
        /// </summary>
        /// <param name="folds">A list of tuples each containing a training set and a validation set.</param>
        /// <param name="parameterRanges">Dictionary of parameters and their ranges to test.</param>
        /// <returns>The best parameter combination along with averaged performance metrics.</returns>
        public (double bestRegularization, double bestIntercept, double[] bestMetrics) CrossValidation(
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
            double bestRegularization = 0;
            double bestIntercept = 0;

            foreach (double regularization in parameterRanges["Regularization"])
            {
                foreach (int intercept in parameterRanges["Intercept"])
                {
                    List<double> accuracies = new List<double>();
                    List<double> precisions = new List<double>();
                    List<double> recalls = new List<double>();
                    List<double> f1Scores = new List<double>();

                    for (int j = 0; j < tempTrainingSetsList.Count; j++)
                    {
                        Train(tempTrainingSetsList[j], regularization, intercept);

                        int[] validationSetPredictions = Predict(tempValidationSetsList[j]);
                        (_, int[] validationSetOutputs) = DataUtility.recordsToInputsOutputs(tempValidationSetsList[j]);


                        (double accuracy, 
                        double precision,
                        double recall,
                        double f1score) = MachineLearningModel.ComputeMetrics(tempValidationSetsList[j],validationSetPredictions);
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
                        bestRegularization = regularization;
                        bestIntercept = intercept;
                    }

                }

                
            }
            double[] bestMetrics = new double[] { bestAccuracy, bestPrecision, bestRecall, bestF1 };

            Console.WriteLine("\nBest parameters after cross validation are:\n");
            Console.WriteLine($"Regularization: {bestRegularization}");
            Console.WriteLine($"Intercept: {bestIntercept}");
            Console.WriteLine("\nTraining set Metrics for best parameters:\n");
            Console.WriteLine($"Accuracy: {bestAccuracy} , Precision: {bestPrecision}");
            Console.WriteLine($"Recall: {bestRecall}, F1 score: {bestF1}");


            return (bestRegularization, bestIntercept, bestMetrics);
            }


        }
    }
