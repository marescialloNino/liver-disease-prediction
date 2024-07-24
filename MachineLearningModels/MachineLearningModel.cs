
using System.Collections.Generic;
using liver_disease_prediction.dataModels;
using System.Linq;
using System;

namespace liver_disease_prediction.MachineLearningModels
{

    /// <summary>
    /// Abstract base class for all machine learning models. 
    /// This class provides a common interface and shared functionality across different types of models.
    /// </summary>
    public abstract class MachineLearningModel
    {

        public abstract int[] Predict(List<LiverPatientRecord> records);


        /// <summary>
        /// Computes accuracy, precision, recall, and F1 score from the predictions made by the model compared to the actual data.
        /// </summary>
        /// <param name="records">A list of LiverPatientRecord instances containing the true data.</param>
        /// <param name="predictions">An array of integer predictions made by the model where 1 indicates presence of disease and 0 indicates absence.</param>
        /// <returns>A tuple containing accuracy, precision, recall, and F1 score as doubles.</returns>
        public static (double accuracy, double precision, double recall, double f1Score) ComputeMetrics(List<LiverPatientRecord> records, int[] predictions)
        {
            int tp = 0; // True Positives
            int tn = 0; // True Negatives
            int fp = 0; // False Positives
            int fn = 0; // False Negatives

            int[] outputs = records.Select(r => r.Dataset).ToArray();

            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == 1 && outputs[i] == 1)
                    tp++;
                else if (predictions[i] == 0 && outputs[i] == 0)
                    tn++;
                else if (predictions[i] == 1 && outputs[i] == 0)
                    fp++;
                else if (predictions[i] == 0 && outputs[i] == 1)
                    fn++;
            }

            double accuracy = (double)(tp + tn) / predictions.Length;
            double precision = tp == 0 ? 0 : (double)tp / (tp + fp);
            double recall = tp == 0 ? 0 : (double)tp / (tp + fn);
            double f1Score = precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

            return (accuracy, precision, recall, f1Score);
        }

    }
}
