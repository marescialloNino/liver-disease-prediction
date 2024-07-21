using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using liver_disease_prediction.dataModels;

namespace liver_disease_prediction.MachineLearningModels
{
    public abstract class MachineLearningModel
    {
        // Abstract method for training the model
        public abstract void Train(List<LiverPatientRecord> records);

        // Abstract method for making predictions
        public abstract int[] Predict(List<LiverPatientRecord> records);

        // Concrete implementation of the Validate method that can be used by all derived classes

        public double ComputeAccuracy(List<LiverPatientRecord> records, int[] predictions)
        {
            int correct = 0;
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == outputs[i])
                {
                    correct++;
                }
            }
            return (double)correct / predictions.Length;
        }
    }
}
