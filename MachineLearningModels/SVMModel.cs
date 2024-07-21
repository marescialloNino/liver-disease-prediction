
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;

namespace liver_disease_prediction.MachineLearningModels
{
    public class SVMModel
    {

        // with kernels is possible to produce non linear boundaries that perfectly separates the data
        public SupportVectorMachine<IKernel> svm;
        


        public void Train(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset == 1 ? 1 : -1).ToArray(); // SVM in Accord.NET expects -1 or 1 for binary classes

            // Set up the learning algorithm
            var learn = new SequentialMinimalOptimization<IKernel>()
            {
                Kernel = new Gaussian(),
                UseComplexityHeuristic = true,
                UseKernelEstimation = true,
            };

            this.svm = learn.Learn(inputs, outputs);
            
        }

        public int[] Predict(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            bool[] predictions = svm.Decide(inputs);  // Returns true for 1 and false for -1
            int[] binaryPredictions = Array.ConvertAll(predictions, p => p ? 1 : 0);
            return binaryPredictions;
        }

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
        
        public void HyperparameterTuning(List<LiverPatientRecord> records)
        {

            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            // Instantiate a new Grid Search algorithm for Kernel Support Vector Machines
            var gridsearch = GridSearch<double[], int>.Create(

                // Here we can specify the range of the parameters to be included in the search
                ranges: new
                {
                    Kernel = GridSearch.Values<IKernel>(new Linear(), new ChiSquare(), new Gaussian()),
                    Complexity = GridSearch.Values(1e-7, 1e-5, 1e-4),
                    Tolerance = GridSearch.Values(1e-5, 1e-2, 1.0)
                },

                // Indicate how learning algorithms for the models should be created
                learner: (p) => new SequentialMinimalOptimization<IKernel>
                {
                    Complexity = p.Complexity,
                    Kernel = p.Kernel.Value,
                    Tolerance = p.Tolerance
                },

                // Define how the model should be learned, if needed
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),

                // Define how the performance of the models should be measured
                loss: (actual, expected, m) => new ZeroOneLoss(expected).Loss(actual)
            );

            // Search for the best model parameters
            var result = gridsearch.Learn(inputs, outputs);

            // Get the best SVM found during the parameter search
            this.svm = result.BestModel;

            // Get an estimate for its error:
            double bestError = result.BestModelError;

            // Get the best values for the parameters:
            double bestC = result.BestParameters.Complexity;
            double bestTolerance = result.BestParameters.Tolerance;
            IKernel bestKernel = result.BestParameters.Kernel.Value;

            Console.WriteLine("SVM HYPERPARAMETER TUNING");
            Console.WriteLine($"BEST PARAMETERS \n : Kernel : {bestKernel} \n Complexity : {bestC}\n Tolerance : {bestTolerance} \n ");
            Console.WriteLine($"BEST ERROR : {bestError}");
        }


    }
}
