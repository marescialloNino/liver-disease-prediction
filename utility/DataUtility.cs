using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System;
using liver_disease_prediction.dataModels;


namespace liver_disease_prediction.utility
{
    public static class DataUtility
    {
        public static List<LiverPatientRecord> LoadDataFromCsv(string path)
        {
            string filePath = path;
            CsvConfiguration config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                MissingFieldFound = null
            };

            using (StreamReader reader = new StreamReader(filePath))
            using (CsvReader csv = new CsvReader(reader, config))
            {
                csv.Context.RegisterClassMap<LiverPatientRecordMap>();
                var records = csv.GetRecords<LiverPatientRecord>();

                return records.ToList();
            }
        }

        // useful for data vizualization
        public static Dictionary<string, List<double>> ExtractFieldDataAsDoubles(List<LiverPatientRecord> recordsList,
                                                                                int? genderFilter = null,
                                                                                int? diseaseFilter = null)

        {
            // Apply filters only if they are not null
            List<LiverPatientRecord> records = recordsList
                .Where(r => (!genderFilter.HasValue || r.Gender == genderFilter.Value) &&
                            (!diseaseFilter.HasValue || r.Dataset == diseaseFilter.Value))
                .ToList();
            Dictionary<string, List<double>> fieldData = new Dictionary<string, List<double>>();

            List<double> ageData = records.Select(r => (double)r.Age).ToList();
            fieldData.Add("Age", ageData);
            List<double> genderData = records.Select(r => (double)r.Gender).ToList();
            fieldData.Add("Gender", genderData);
            List<double> totalBilirubinData = records.Select(r => r.TotalBilirubin).ToList();
            fieldData.Add("TotalBilirubin", totalBilirubinData);
            List<double> directBilirubinData = records.Select(r => r.DirectBilirubin).ToList();
            fieldData.Add("DirectBilirubin", directBilirubinData);
            List<double> alkalinePhosphotaseData = records.Select(r => (double)r.AlkalinePhosphotase).ToList();
            fieldData.Add("AlkalinePhosphotase", alkalinePhosphotaseData);
            List<double> alamineAminotransferaseData = records.Select(r => (double)r.AlamineAminotransferase).ToList();
            fieldData.Add("AlamineAminotransferase", alamineAminotransferaseData);
            List<double> aspartateAminotransferaseData = records.Select(r => (double)r.AspartateAminotransferase).ToList();
            fieldData.Add("AspartateAminotransferase", aspartateAminotransferaseData);
            List<double> totalProteinsData = records.Select(r => r.TotalProtiens).ToList();
            fieldData.Add("TotalProtiens", totalProteinsData);
            List<double> albuminData = records.Select(r => r.Albumin).ToList();
            fieldData.Add("Albumin", albuminData);
            List<double> albuminAndGlobulinRatioData = records.Select(r => r.AlbuminAndGlobulinRatio).ToList();
            fieldData.Add("AlbuminAndGlobulinRatio", albuminAndGlobulinRatioData);
            List<double> datasetData = records.Select(r => (double)r.Dataset).ToList();
            fieldData.Add("Dataset", datasetData);

            return fieldData;
        }

        public static (double[][], int[]) recordsToInputsOutputs(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            return (inputs, outputs);
        }

        public static (List<LiverPatientRecord>, List<LiverPatientRecord>) SplitData(List<LiverPatientRecord> data, double trainSize = 0.8)
        {

            Random random = new Random();
            // Shuffle the list randomly
            var shuffledData = data.OrderBy(x => random.Next()).ToList();

            // Calculate split index
            int splitIndex = (int)(trainSize * shuffledData.Count);

            // Create training and testing lists
            List<LiverPatientRecord> trainingSet = shuffledData.Take(splitIndex).ToList();
            List<LiverPatientRecord> testingSet = shuffledData.Skip(splitIndex).ToList();

            return (trainingSet, testingSet);
        }


        public static double[][] ScaleFeatures(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int numFeatures = inputs[0].Length;

            // Assume gender is at index 1, adjust if different
            int genderIndex = 1;

            double[] means = new double[numFeatures];
            double[] stdDevs = new double[numFeatures];

            // Calculate means and standard deviations, skipping binary gender
            for (int j = 0; j < numFeatures; j++)
            {
                if (j != genderIndex)
                {
                    means[j] = inputs.Average(row => row[j]);
                    stdDevs[j] = Math.Sqrt(inputs.Sum(row => Math.Pow(row[j] - means[j], 2)) / inputs.Length);
                }
            }

            // Scale features, except for the gender
            double[][] scaledInputs = inputs.Select(row =>
                row.Select((value, index) =>
                    index == genderIndex ? value : (value - means[index]) / stdDevs[index]
                ).ToArray()
            ).ToArray();

            return scaledInputs;
        }


    }
}
