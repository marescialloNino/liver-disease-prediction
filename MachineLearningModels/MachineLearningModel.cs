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
        public abstract bool[] Predict(List<LiverPatientRecord> records);

        // Concrete implementation of the Validate method that can be used by all derived classes
        public virtual double Validate(List<LiverPatientRecord> records, bool[] predictions)
        {
            int correctPredictions = 0;
            for (int i = 0; i < records.Count; i++)
            {
                if ((predictions[i] && records[i].Dataset == 1) || (!predictions[i] && records[i].Dataset == 0))
                {
                    correctPredictions++;
                }
            }
            return (double)correctPredictions / records.Count;
        }
    }
}
