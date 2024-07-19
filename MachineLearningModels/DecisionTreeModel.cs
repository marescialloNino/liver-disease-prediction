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

namespace liver_disease_prediction.MachineLearningModels
{


    public class DecisionTreeModel
    {
        private DecisionTree tree;
        private DecisionVariable[] features = new DecisionVariable[]
        {
            new DecisionVariable("Age", DecisionVariableKind.Continuous),
            new DecisionVariable("Gender", 2), 
            new DecisionVariable("Direct Bilirubin", DecisionVariableKind.Continuous),
            new DecisionVariable("Alkaline Phosphatase", DecisionVariableKind.Continuous),
            new DecisionVariable("Aspartate Aminotransferase", DecisionVariableKind.Continuous),
            new DecisionVariable("Total Protiens", DecisionVariableKind.Continuous),
            new DecisionVariable("Albumin and Globulin Ratio", DecisionVariableKind.Continuous)
        };
        
        public DecisionTreeModel()
        {
            this.tree = new DecisionTree(features, 2);
        }

        public void Train(List<LiverPatientRecord> records)
        {
            // Create an instance of the C4.5 learning algorithm
            C45Learning teacher = new C45Learning(features);

            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();

            // Use the learning algorithm to induce the tree
            tree = teacher.Learn(inputs, outputs);
        }

        public int[] Predict(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            return tree.Decide(inputs);
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
