using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace OpenCognition.Models.SentimentPack
{
    public class SentimentData
    {


        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
        //1 is positive, -1 is negative
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]

        public float Probability { get; set; }

        public float Score { get; set; }
    }


}
