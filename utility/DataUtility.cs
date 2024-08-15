
using liver_disease_prediction.dataModels;
using System.Collections.Generic;
using System.Linq;
using System;
using System.IO;

namespace liver_disease_prediction.utility
{
    public static class DataUtility
    {

        /// <summary>
        /// Reads a CSV file containing liver patient records and returns a list of records.
        /// </summary>
        /// <param name="path">The file path of the CSV file.</param>
        /// <returns>A List of LiverPatientRecord objects populated from the CSV data.</returns>
        public static List<LiverPatientRecord> LoadDataFromCsv(string path)
        {
            var records = new List<LiverPatientRecord>();
            string line;
            using (var reader = new StreamReader(path))
            {
                // Read header line if there is one
                reader.ReadLine();

                while ((line = reader.ReadLine()) != null)
                {
                    string[] values = line.Split(',');

                    LiverPatientRecord record = new LiverPatientRecord
                    {
                        Age = int.Parse(values[0]),
                        Gender = ConvertGender(values[1]),
                        TotalBilirubin = double.Parse(values[2]),
                        DirectBilirubin = double.Parse(values[3]),
                        AlkalinePhosphotase = int.Parse(values[4]),
                        AlamineAminotransferase = int.Parse(values[5]),
                        AspartateAminotransferase = int.Parse(values[6]),
                        TotalProtiens = double.Parse(values[7]),
                        Albumin = double.Parse(values[8]),
                        AlbuminAndGlobulinRatio = string.IsNullOrEmpty(values[9]) ? 0.95 : double.Parse(values[9]),
                        Dataset = ConvertDataset(values[10])
                    };
                    records.Add(record);
                }
            }
            return records;
        }


        /// <summary>
        /// Converts a gender string to its corresponding integer value.
        /// </summary>
        /// <param name="gender">The gender as a string.</param>
        /// <returns>The integer representation of the gender (0: "male" , 1: "female").</returns>
        private static int ConvertGender(string gender)
        {
            string normalizedString = gender.Trim().ToLowerInvariant();

            if(normalizedString == "male")
            {
                return 0;
            }
            if (normalizedString == "female")
            {
                return 1;
            }
            else
            {
                throw new ArgumentException("Invalid gender value");
            }
        }

        /// <summary>
        /// Converts dataset strings into integers, specifically mapping '2' to '0' and '1' to '1'.
        /// </summary>
        /// <param name="dataset">The dataset value as a string.</param>
        /// <returns>The mapped integer value of the dataset.</returns>
        private static int ConvertDataset(string dataset)
        {
            if (int.Parse(dataset) == 1)
            {
                return 1;
            }
            if (int.Parse(dataset) == 2)
            {
                return 0;
            }
            else
            {
                throw new ArgumentException("Invalid dataset value");
            }
        }

        /// <summary>
        /// Extracts and organizes feature data from liver patient records into a dictionary.
        /// This helps in managing data for visualization and analysis.
        /// </summary>
        /// <param name="recordsList">List of LiverPatientRecord objects.</param>
        /// <param name="genderFilter">Value of the gender field for which the records are filtered (0:male, 1:female) (default = null).</param>
        /// <param name="diseaseFilter">Value of the dataset field for which the records are filtered (0:no disease, 1:disease) (default = null).</param>
        /// <returns>Dictionary with keys as feature names and values as lists of feature data.</returns>
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

        /// <summary>
        /// Scales and normalizes the feature data of liver patient records for visualization.
        /// </summary>
        /// <param name="records">List of LiverPatientRecord objects to process.</param>
        /// <returns>Dictionary with feature names as keys and scaled, normalized list of doubles as values.</returns>
        public static Dictionary<string, List<double>> GetScaledAndNormalizedData(List<LiverPatientRecord> records)
        {
            Dictionary<string, List<double>> fieldData = ExtractFieldDataAsDoubles(records);
            Dictionary<string, List<double>> normalizedData = new Dictionary<string, List<double>>();

            foreach (var entry in fieldData)
            {
                double mean = entry.Value.Average();
                double stdDev = Math.Sqrt(entry.Value.Sum(x => Math.Pow(x - mean, 2)) / entry.Value.Count);
                List<double> normalizedValues = entry.Value.Select(x => (x - mean) / stdDev).ToList();
                normalizedData.Add(entry.Key, normalizedValues);
            }

            return normalizedData;
        }


        /// <summary>
        /// Converts a list of LiverPatientRecord objects to feature arrays and output arrays.
        /// </summary>
        /// <param name="records">List of LiverPatientRecord objects.</param>
        /// <returns>Tuple containing arrays of feature inputs and corresponding output labels.</returns>
        public static (double[][], int[]) recordsToInputsOutputs(List<LiverPatientRecord> records)
        {
            double[][] inputs = records.Select(r => r.SelectedFeaturesArray()).ToArray();
            int[] outputs = records.Select(r => r.Dataset).ToArray();
            return (inputs, outputs);
        }


        /// <summary>
        /// Splits data into a training set and a testing set based on a specified training size proportion.
        /// </summary>
        /// <param name="data">List of LiverPatientRecord objects to be split.</param>
        /// <param name="trainSize">Proportion of the data to be used for training (default is 0.8).</param>
        /// <returns>Tuple of training set and testing set.</returns>
        public static (List<LiverPatientRecord>, List<LiverPatientRecord>) SplitData(List<LiverPatientRecord> data, double trainSize = 0.8)
        {
            Random random = new Random();
            // Shuffle the list randomly
            List<LiverPatientRecord> shuffledData = data.OrderBy(x => random.Next()).ToList();

            // Calculate split index
            int splitIndex = (int)(trainSize * shuffledData.Count);

            // Create training and testing lists
            List<LiverPatientRecord> trainingSet = shuffledData.Take(splitIndex).ToList();
            List<LiverPatientRecord> testingSet = shuffledData.Skip(splitIndex).ToList();

            return (trainingSet, testingSet);
        }

        /// <summary>
        /// Splits data into a specified number of folds for k-fold cross-validation.
        /// </summary>
        /// <param name="data">List of LiverPatientRecord objects to be split into folds.</param>
        /// <param name="k">The number of folds to create (default is 5).</param>
        /// <returns>List of folds, where each fold is a list of LiverPatientRecord objects.</returns>
        public static List<List<LiverPatientRecord>> SplitDataIntoFolds(List<LiverPatientRecord> data, int k = 5)
        {
            Random random = new Random();
            // Shuffle the list randomly
            List<LiverPatientRecord> shuffledData = data.OrderBy(x => random.Next()).ToList();

            // Create the list of folds
            List<List<LiverPatientRecord>> folds = new List<List<LiverPatientRecord>>();

            // Calculate the number of elements per fold
            int totalItems = data.Count;
            int baseFoldSize = (int)totalItems / k;
            int remainder = totalItems % k;

            int start = 0;
            for (int i = 0; i < k; i++)
            {
                int foldSize = baseFoldSize + (remainder-- > 0 ? 1 : 0); // Distribute the remainder among the first few folds
                if (start + foldSize > totalItems)
                {
                    foldSize = totalItems - start; // Adjust fold size to prevent going out of bounds
                }

                // Allocate the current fold
                List<LiverPatientRecord> currentFold = shuffledData.GetRange(start, foldSize);
                folds.Add(currentFold);
                start += foldSize; // Move the start pointer
            }

            return folds;
        }

        /// <summary>
        /// Scales, normalizes, and caps outliers in the dataset.
        /// </summary>
        /// <param name="inputs">Array of feature arrays, where each inner array corresponds to a specific feature across all records.</param>
        /// <returns>A new double[][] array where each feature has been scaled, normalized, and had its outliers capped.</returns>
        public static double[][] PreprocessFeatures(double[][] inputs)
        {
            int numFeatures = inputs[0].Length;  
            double[][] scaledInputs = new double[inputs.Length][];

            for (int j = 0; j < numFeatures; j++)
            {
                double[] featureData = inputs.Select(x => x[j]).ToArray();
                double mean = featureData.Average();
                double stdDev = Math.Sqrt(featureData.Sum(x => Math.Pow(x - mean, 2)) / featureData.Length);

                // Normalize feature data
                double[] normalizedData = featureData.Select(x => (x - mean) / stdDev).ToArray();

                // Calculate percentiles for outlier detection
                double q1 = StatisticsUtility.CalculatePercentile(normalizedData.ToList(), 25);
                double q3 = StatisticsUtility.CalculatePercentile(normalizedData.ToList(), 75);
                double iqr = q3 - q1;
                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;

                // Cap outliers
                double[] adjustedData = normalizedData.Select(x => Math.Min(Math.Max(x, lowerBound), upperBound)).ToArray();

                // Assign adjusted data back to the scaled inputs
                for (int i = 0; i < inputs.Length; i++)
                {
                    if (scaledInputs[i] == null)
                        scaledInputs[i] = new double[numFeatures];

                    scaledInputs[i][j] = adjustedData[i];
                }
            }

            return scaledInputs;
        }


    }
}
