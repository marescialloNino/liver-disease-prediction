
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning;

namespace liver_disease_prediction.MachineLearningModels
{
    public class SVMModel
    {
        private SupportVectorMachine<Gaussian> svm;

        public void Train(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset == 1 ? 1 : -1).ToArray(); // SVM in Accord.NET expects -1 or 1 for binary classes

            // Set up the learning algorithm
            var learn = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = true,
            };

            svm = learn.Learn(inputs, outputs);
        }

        public int[] Predict(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            bool[] predictions = svm.Decide(inputs);  // Returns true for 1 and false for -1
            int[] binaryPredictions = Array.ConvertAll(predictions, p => p ? 1 : 0);
            return binaryPredictions;
        }

        public double Validate(List<LiverPatientRecord> records, int[] predictions)
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
