using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using liver_disease_prediction.dataModels;


namespace liver_disease_prediction.utility
{
    public static class DataUtility
    {

        /// <summary>
        /// Loads liver patient records from a CSV file.
        /// </summary>
        /// <param name="path">Path to the CSV file.</param>
        /// <returns>List of LiverPatientRecord objects populated from the CSV data.</returns>
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
                List<LiverPatientRecord> records = csv.GetRecords<LiverPatientRecord>().ToList();

                return records;
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


    }
}
