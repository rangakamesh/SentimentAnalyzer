using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OpenCognition.Models.SentimentPack
{
    public class SentimentCReturn
    {
        public bool prediction;
        public int predictionVal;
        public float predictionConfidence;

        public SentimentCReturn(bool p, float pC)
        {
            prediction = p;

            if (prediction)
            {
                predictionVal = 1;
            }
            else
            {
                predictionVal = -1;
            }

            predictionConfidence = pC;
        }
    }
}
