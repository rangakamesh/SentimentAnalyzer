using System;
using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OpenCognition.Models.SentimentPack
{
    public class VoteClassifiers
    {

        public List<PredictionEngine<SentimentData, SentimentPrediction>> predEngines;

        MLContext mlContext = new MLContext();


        public VoteClassifiers(List<ITransformer> classifierList)
        {
            predEngines = new List<PredictionEngine<SentimentData, SentimentPrediction>>();

            foreach (ITransformer model in classifierList)
            {
                predEngines.Add(mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model));
            }
        }

        public bool singularPredict(SentimentData predictInput, ref float confidence)
        {
            List<bool> decisions = new List<bool>();

            foreach (PredictionEngine<SentimentData, SentimentPrediction> predEngine in predEngines)
            {
                SentimentPrediction pred = predEngine.Predict(predictInput);
                decisions.Add(pred.Prediction);
            }

            bool finalDecision = decisions.GroupBy(decision => decision).OrderByDescending(decx => decx.Count()).First().Key;
            float finalDecisionCount = decisions.GroupBy(decision => decision).OrderByDescending(decx => decx.Count()).First().Count();
            confidence = finalDecisionCount / decisions.Count();

            return finalDecision;
        }
    }
}
