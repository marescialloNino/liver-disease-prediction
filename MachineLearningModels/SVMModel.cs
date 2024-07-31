
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using liver_disease_prediction.dataModels;
using liver_disease_prediction.utility;
using System.Collections.Generic;
using System.Linq;
using System;

namespace liver_disease_prediction.MachineLearningModels
{
    public class SVMModel
    {
        public SupportVectorMachine<IKernel> Svm {  get; set; }


        /// <summary>
        /// Trains the SVM model using specified kernel and complexity parameter.
        /// </summary>
        /// <param name="records">List of liver patient records to train the model.</param>
        /// <param name="kernel">The kernel function to use in the SVM.</param>
        /// <param name="complexity">
        /// The complexity parameter C, which helps in controlling the trade-off between 
        /// achieving a low error on the training data and minimizing the norm of the weights.
        /// </param>
        public void Train(List<LiverPatientRecord> records, IKernel kernel, double complexity)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            double[][] preprocessedInputs = DataUtility.PreprocessFeatures(inputs);
            int[] outputs = records.Select(r => r.Dataset == 1 ? 1 : -1).ToArray(); // SVM in Accord.NET expects -1 or 1 for binary classes

            // Set up the learning algorithm
            SequentialMinimalOptimization<IKernel> teacher = new SequentialMinimalOptimization<IKernel>()
            {
                Kernel = kernel,
                Complexity = complexity

            };     

            Svm = teacher.Learn(preprocessedInputs, outputs);

        }

        /// <summary>
        /// Predicts disease presence for a list of liver patient records using the trained SVM model.
        /// </summary>
        /// <param name="records">The records to predict.</param>
        /// <returns>An array of predictions where 1 indicates presence of disease and -1 indicates absence.</returns>
        public int[] Predict(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            bool[] predictions = Svm.Decide(inputs);  // Returns true for 1 and false for -1
            int[] binaryPredictions = Array.ConvertAll(predictions, p => p ? 1 : 0);
            return binaryPredictions;
        }


        /// <summary>
        /// Performs k-fold cross-validation with hyperparameter tuning and evaluates model performance.
        /// </summary>
        /// <param name="foldedTrainSet">A list of lists each containing the k-th fold of the training set.</param>
        /// <param name="kernelRange">Array of different kernels to be tested during cross-validation.</param>
        /// <param name="complexityRange">Array of different complexity values to be tested during cross-validation.</param>
        /// <returns>The best kernel and complexity values along with the averaged performance metrics (accuracy, precision, recall, F1 score).</returns>
        public ( IKernel bestKernel, double bestComplexity,double[] bestMetrics) CrossValidation(
            List<List<LiverPatientRecord>> foldedTrainSet, IKernel[] kernelRange, double[] complexityRange)
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
            IKernel bestKernel = new Linear();
            double bestComplexity = 0.0;
            

            foreach (IKernel kernel in kernelRange)
            {
                
                
                    foreach (double complexity in complexityRange)
                    {
                        List<double> accuracies = new List<double>();
                        List<double> precisions = new List<double>();
                        List<double> recalls = new List<double>();
                        List<double> f1Scores = new List<double>();

                        for (int j = 0; j < tempTrainingSetsList.Count; j++)
                        {
                            Train(tempTrainingSetsList[j], kernel, complexity);

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
                            bestKernel = kernel;
                            bestComplexity = complexity;
                            
                        }

                    }
                


            }
            double[] bestMetrics = new double[] { bestAccuracy, bestPrecision, bestRecall, bestF1 };

            Console.WriteLine("\nBest parameters after cross validation are:\n");
            Console.WriteLine($"Kernel: {bestKernel}");
            Console.WriteLine($"Complexity: {bestComplexity}");
            
            Console.WriteLine("\nTraining set Metrics for best parameters:\n");
            Console.WriteLine($"Accuracy: {bestAccuracy} , Precision: {bestPrecision}");
            Console.WriteLine($"Recall: {bestRecall}, F1 score: {bestF1}");


            return (bestKernel, bestComplexity, bestMetrics);
        }

    }
}
