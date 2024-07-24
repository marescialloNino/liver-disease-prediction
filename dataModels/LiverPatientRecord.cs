using System;

namespace liver_disease_prediction.dataModels
{
    public class LiverPatientRecord
    {
        public int Age { get; set; }
        public int Gender { get; set; }
        public double TotalBilirubin { get; set; }
        public double DirectBilirubin { get; set; }
        public int AlkalinePhosphotase { get; set; }
        public int AlamineAminotransferase { get; set; }
        public int AspartateAminotransferase { get; set; }
        public double TotalProtiens { get; set; }
        public double Albumin { get; set; }
        public double AlbuminAndGlobulinRatio { get; set; } 
        public int Dataset { get; set; }

        // Convert to array of the selected features (chosen via previous data exploration) excluding the label "Dataset"
        public double[] SelectedFeaturesArray()
        {
            return new double[] { Age, Gender, DirectBilirubin, AlkalinePhosphotase, AspartateAminotransferase, TotalProtiens, AlbuminAndGlobulinRatio };
        }
    }
}
