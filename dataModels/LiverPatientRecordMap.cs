
using CsvHelper.Configuration;
using CsvHelper.TypeConversion;
using CsvHelper;
using System;

namespace liver_disease_prediction.dataModels
{
    // This class is needed to map the csv rows into the LiverPatientRecord class with the right naming
    // convention. Handling missing values in the record.

    public class LiverPatientRecordMap : ClassMap<LiverPatientRecord>
    {
        public LiverPatientRecordMap()
        {
            Map(m => m.Age).Index(0);
            Map(m => m.Gender).Index(1).TypeConverter<GenderToIntConverter>(); 
            Map(m => m.TotalBilirubin).Index(2);
            Map(m => m.DirectBilirubin).Index(3);
            Map(m => m.AlkalinePhosphotase).Index(4);
            Map(m => m.AlamineAminotransferase).Index(5);
            Map(m => m.AspartateAminotransferase).Index(6);
            Map(m => m.TotalProtiens).Index(7);
            Map(m => m.Albumin).Index(8);
            Map(m => m.AlbuminAndGlobulinRatio).Index(9).Default(0.95); // handle missing values
            Map(m => m.Dataset).Index(10).TypeConverter<DatasetToIntConverter>();
            
        }
    }

    public class GenderToIntConverter : ITypeConverter
    {
        public object ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            // Normalize the text to ensure consistency
            var normalized = text?.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "male":
                    return 0;
                case "female":
                    return 1;
                default:
                    throw new ArgumentException("Invalid gender value"); // Or return a default value
            }
        }

        public string ConvertToString(object value, IWriterRow row, MemberMapData memberMapData)
        {
            // Convert the integers back to string if needed, for writing CSVs
            return value switch
            {
                0 => "male",
                1 => "female",
                _ => throw new ArgumentException("Invalid gender value")
            };
        }
    }

    public class DatasetToIntConverter : ITypeConverter
    {
        public object ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            if (int.TryParse(text, out int result))
            {
                // Map '2' (no disease) to '0' and '1' (has disease) to '1'
                return result == 2 ? 0 : 1;
            }
            else
            {
                throw new ArgumentException("Invalid dataset value");
            }
        }

        public string ConvertToString(object value, IWriterRow row, MemberMapData memberMapData)
        {
            // Convert the integers back to string if needed, for writing CSVs
            return value switch
            {
                0 => "2", // Originally 'no disease'
                1 => "1", // Originally 'has disease'
                _ => throw new ArgumentException("Invalid dataset value")
            };
        }
    }
}
