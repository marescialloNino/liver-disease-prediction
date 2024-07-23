using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using liver_disease_prediction.dataModels;

namespace liver_disease_prediction.utility
{
    public static class StatisticsUtility
    {

        /// <summary>
        /// Calculates the mean of a list of values.
        /// </summary>
        /// <param name="values">List of double values.</param>
        /// <returns>The mean of the list or 0 if the list is empty.</returns>
        public static double CalculateMean(List<double> values)
        {
            return values.Count == 0 ? 0 : values.Average();
        }


        /// <summary>
        /// Calculates the median of a list of values.
        /// </summary>
        /// <param name="values">List of double values.</param>
        /// <returns>The median of the list or 0 if the list is empty.</returns>
        public static double CalculateMedian(List<double> values)
        {
            if (values.Count == 0)
                return 0;

            List<double> sortedValues = values.OrderBy(n => n).ToList();
            int middle = sortedValues.Count / 2;
            if (sortedValues.Count % 2 == 0)
            {
                return (sortedValues[middle] + sortedValues[middle - 1]) / 2.0;
            }
            else
            {
                return sortedValues[middle];
            }
        }


        /// <summary>
        /// Calculates the standard deviation of a list of values.
        /// </summary>
        /// <param name="values">List of double values.</param>
        /// <returns>The standard deviation of the list or 0 if the list is empty.</returns>
        public static double CalculateStandardDeviation(List<double> values)
        {
            if (values.Count == 0)
                return 0;

            double avg = CalculateMean(values);
            double sum = values.Sum(d => Math.Pow(d - avg, 2));
            return Math.Sqrt(sum / values.Count);
        }


        /// <summary>
        /// Calculates the correlation matrix for fields from liver patient records.
        /// </summary>
        /// <param name="records">List of LiverPatientRecord object.</param>
        /// <returns>A 2D array representing the correlation matrix.</returns>
        public static double[,] CalculateCorrelationMatrixForLiverPatientRecords(List<LiverPatientRecord> records)
        {

            Dictionary<string, List<double>> recordsDict =  DataUtility.ExtractFieldDataAsDoubles(records);
            List<List<double>> columns = new List<List<double>>
        {
            recordsDict["Age"], recordsDict["Gender"],recordsDict["TotalBilirubin"],recordsDict["DirectBilirubin"],recordsDict["AlkalinePhosphotase"],
            recordsDict["AlamineAminotransferase"],recordsDict["AspartateAminotransferase"],recordsDict["TotalProtiens"],recordsDict["Albumin"],recordsDict["AlbuminAndGlobulinRatio"]
        };

            return CalculateCorrelationMatrix(columns);
        }


        /// <summary>
        /// Calculates the correlation matrix from a list of columns of data.
        /// </summary>
        /// <param name="columns">List of lists, each containing data for a specific variable.</param>
        /// <returns>A 2D array representing the correlation matrix.</returns>
        public static double[,] CalculateCorrelationMatrix(List<List<double>> columns)
        {
            int n = columns.Count;
            double[,] correlationMatrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    if (i == j)
                    {
                        correlationMatrix[i, j] = 1.0;  // Correlation with itself is always 1
                    }
                    else
                    {
                        double corr = CalculateCorrelation(columns[i], columns[j]);
                        correlationMatrix[i, j] = corr;
                        correlationMatrix[j, i] = corr;  // Correlation matrix is symmetric
                    }
                }
            }

            return correlationMatrix;
        }


        /// <summary>
        /// Calculates the correlation coefficient between two lists of values.
        /// </summary>
        /// <param name="x">First list of double values.</param>
        /// <param name="y">Second list of double values.</param>
        /// <returns>The correlation coefficient between the two lists.</returns>
        public static double CalculateCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count == 0)
                return 0;

            double meanX = x.Average();
            double meanY = y.Average();
            double sumXY = 0;
            double sumX2 = 0;
            double sumY2 = 0;

            for (int i = 0; i < x.Count; i++)
            {
                double xi = x[i] - meanX;
                double yi = y[i] - meanY;
                sumXY += xi * yi;
                sumX2 += xi * xi;
                sumY2 += yi * yi;
            }

            double stdX = Math.Sqrt(sumX2 / x.Count);
            double stdY = Math.Sqrt(sumY2 / y.Count);
            double covariance = sumXY / x.Count;

            return covariance / (stdX * stdY);
        }
    } 
}
