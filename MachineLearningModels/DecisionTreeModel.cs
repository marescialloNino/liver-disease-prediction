using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Statistics.Filters;
using Accord.MachineLearning.DecisionTrees.Rules;
using System.Threading.Tasks;
using liver_disease_prediction.dataModels;
using Accord.MachineLearning;
using Accord.MachineLearning.Performance;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using liver_disease_prediction.utility;

namespace liver_disease_prediction.MachineLearningModels
{


    public class DecisionTreeModel: MachineLearningModel
    {
        private DecisionTree tree {  get; set; }
        private DecisionVariable[] features {  get; set; }
        
        
        public DecisionTreeModel()
        {
            this.features = new DecisionVariable[]
                            {
                                new DecisionVariable("Age", DecisionVariableKind.Continuous),
                                new DecisionVariable("Gender", 2),
                                new DecisionVariable("Direct Bilirubin", DecisionVariableKind.Continuous),
                                new DecisionVariable("Alkaline Phosphatase", DecisionVariableKind.Continuous),
                                new DecisionVariable("Aspartate Aminotransferase", DecisionVariableKind.Continuous),
                                new DecisionVariable("Total Protiens", DecisionVariableKind.Continuous),
                                new DecisionVariable("Albumin and Globulin Ratio", DecisionVariableKind.Continuous)
                            };
            this.tree = new DecisionTree(features, 2);
        }

        public override void Train(List<LiverPatientRecord> records)
        {
            // Create an instance of the C4.5 learning algorithm
            C45Learning teacher = new C45Learning(features);

            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);


            // Use the learning algorithm to induce the tree
            tree = teacher.Learn(inputs, outputs);
        }

        public override int[] Predict(List<LiverPatientRecord> records)
        {
            (double[][] inputs, _ ) = DataUtility.recordsToInputsOutputs(records);
            return tree.Decide(inputs);
        }

        public void HyperparameterTuning(List<LiverPatientRecord> records)
        {

            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);
            // Instantiate a new Grid Search algorithm for Kernel Support Vector Machines
            var gridsearch = new GridSearch<DecisionTree, double[], int>()
            {
                // Here we can specify the range of the parameters to be included in the search
                ParameterRanges = new GridSearchRangeCollection()
                {
                new GridSearchRange("join", new double[] { 1,3,5,7,9,11,13,15,17,19}),
                new GridSearchRange("maxHeight", new double[] { 1, 5, 10, 15, 20, 30, 50})
                },

                // Indicate how learning algorithms for the models should be created
                Learner = (p) => new C45Learning(features)
                {
                    Join = (int)p["join"],
                    MaxHeight = (int)p["maxHeight"]
                },
                // Define how the performance of the models should be measured
                Loss = (actual, expected, m) => new ZeroOneLoss(expected).Loss(actual)

            };


            // Search for the best model parameters
            var result = gridsearch.Learn(inputs, outputs);

            // Get the best SVM found during the parameter search
            this.tree = result.BestModel;

            // Get an estimate for its error:
            double bestError = result.BestModelError;

            // Get the best values found for the model parameters:
            double bestJoin = result.BestParameters["join"].Value;
            double bestMaxHeight = result.BestParameters["maxHeight"].Value;

            Console.WriteLine("DECISION TREE HYPERPARAMETER TUNING");
            Console.WriteLine($"BEST PARAMETERS : \n Join : {bestJoin}\n max height : {bestMaxHeight} \n ");
            Console.WriteLine($"BEST ERROR : {bestError}");
        }

        public CrossValidationResult<DecisionTree, double[], int> CrossValidation(List<LiverPatientRecord> records, int folds)
        {
            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);

            var crossvalidation = new CrossValidation<DecisionTree, double[]>()
            {
                K = folds, 

                // Indicate how learning algorithms for the models should be created
                Learner = (s) => new C45Learning(features),

                // Indicate how the performance of those models will be measured
                Loss = (expected, actual, p) => new ZeroOneLoss(expected).Loss(actual),

                Stratify = false, // do not force balancing of classes
            };


            // Compute the cross-validation
            CrossValidationResult<DecisionTree, double[], int> result = crossvalidation.Learn(inputs, outputs);

            return result;
        }


        public GeneralConfusionMatrix GenerateConfusionMatrix(List<LiverPatientRecord> records, 
                                                              CrossValidationResult<DecisionTree, double[], int> crossValRes)
        {
            (double[][] inputs, int[] outputs) = DataUtility.recordsToInputsOutputs(records);
            GeneralConfusionMatrix gcm = crossValRes.ToConfusionMatrix(inputs, outputs);
            return gcm;
        }
        

    }
}
