using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning;
using Accord.Statistics.Analysis;

namespace liver_disease_prediction.MachineLearningModels
{
    public class LogisticRegressionModel
    {
        private LogisticRegression logisticRegression;

        public LogisticRegressionModel() {
            logisticRegression = new LogisticRegression();
        }

        public void Train(List<LiverPatientRecord> records)
        {

            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();

            var teacher = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                Tolerance = 1e-4,  
                MaxIterations = 100,  
                Regularization = 1e-5
                
            };

            logisticRegression = teacher.Learn(inputs, outputs);         

            // After training, inspect the coefficients
            Console.WriteLine("Coefficients:");
            for (int i = 0; i < logisticRegression.Coefficients.Length; i++)
            {
                Console.WriteLine($"Coefficient for input {i}: {logisticRegression.Coefficients[i]}");
            }

            
            Console.WriteLine($"Has Converged? : {teacher.HasConverged}");
        }

        public int[] Predict(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            bool[] predictions = logisticRegression.Decide(inputs);
            int[] binaryPredictions = Array.ConvertAll(predictions, p => p ? 1 : 0);
            return binaryPredictions;

        }

        public CrossValidationResult<LogisticRegression, double[], int> CrossValidation(List<LiverPatientRecord> records, int folds)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();

            var crossvalidation = new CrossValidation<LogisticRegression, double[]>()
            {
                K = folds,

                // Indicate how learning algorithms for the models should be created
                Learner = (s) => new IterativeReweightedLeastSquares<LogisticRegression>()
                {
                    Tolerance = 1e-4,
                    MaxIterations = 100,
                    Regularization = 1e-5

                },

                // Indicate how the performance of those models will be measured
                Loss = (expected, actual, p) => new ZeroOneLoss(expected).Loss(actual),

                Stratify = false, // do not force balancing of classes
            };


            // Compute the cross-validation
            CrossValidationResult<LogisticRegression, double[], int> result = crossvalidation.Learn(inputs, outputs);

            return result;
        }


        public GeneralConfusionMatrix GenerateConfusionMatrix(List<LiverPatientRecord> records,
                                                              CrossValidationResult<LogisticRegression, double[], int> crossValRes)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            GeneralConfusionMatrix gcm = crossValRes.ToConfusionMatrix(inputs, outputs);
            return gcm;
        }

        public double Validate(List<LiverPatientRecord> records, int[] predictions)
        {
            int i = 0;
            double correctPredictions = 0;
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            foreach (LiverPatientRecord record in records)
            {
                if (outputs[i] == predictions[i])
                    correctPredictions++;
                i++;
            }
            return (double)correctPredictions / records.Count;
        }

    }
}
