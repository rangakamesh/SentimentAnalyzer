using Microsoft.ML;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;

namespace OpenCognition.Models.SentimentPack
{
    
    public class SentimentClassifier
    {
        private readonly MLContext mlContext;
        private readonly VoteClassifiers vc;

        List<string> positiveDictionary, negativeDictionary;
        ILogger logger;
        public SentimentClassifier(ILogger log) 
        {
            logger=log;
            logger.LogInformation("New Sentiment Classifier initiated.");

            var positiveFile = File.ReadAllLines("Models/SentimentPack/positiveDictionary.txt");
            // var positiveFile = File.ReadAllLines("positiveDictionary.txt");
            positiveDictionary = new List<string>(positiveFile);

            var negativeFile = File.ReadAllLines("Models/SentimentPack/negativeDictionary.txt");
            // var negativeFile = File.ReadAllLines("negativeDictionary.txt");
            negativeDictionary = new List<string>(negativeFile);

            mlContext = new MLContext();

            DataViewSchema aSchema;
            var amazonModel = mlContext.Model.Load("MLModels/SentimentClassifiers/amazonModel.zip", out aSchema);

            DataViewSchema ySchema;
            var yelpModel = mlContext.Model.Load("MLModels/SentimentClassifiers/yelpModel.zip", out ySchema);

            DataViewSchema iSchema;
            var imdbModel = mlContext.Model.Load("MLModels/SentimentClassifiers/imdbModel.zip", out iSchema);

            List<ITransformer> vcList = new List<ITransformer>() { amazonModel, yelpModel, imdbModel };
            vc = new VoteClassifiers(vcList);

        }

        public SentimentCReturn predict(string text)
        {            
            logger.LogInformation("Predicting sentiment for text : {0}",text);

            List<string> tokenized = new List<string>();
            splitAndCombine(ref tokenized, text);

            SentimentData predictInput = new SentimentData
            {
                SentimentText = text
            };

            float decisionConfidence = 0.0f;
            var decision = vc.singularPredict(predictInput, ref decisionConfidence);

            var returnValue = new SentimentCReturn(decision,decisionConfidence);

            if (returnValue.prediction == false || returnValue.predictionConfidence < 0.70f)
            {
                int posCount = 0, negCount = 0;

                foreach (string s in tokenized)
                {
                    if (positiveDictionary.Any(sx => sx == s))
                    {
                        posCount += 1;
                    }

                    if (negativeDictionary.Any(sx => sx == s))
                    {
                        negCount += 1;
                    }
                }

                if (posCount > negCount)
                {
                    returnValue.prediction = true;
                    returnValue.predictionVal = 1;
                }
                else if (negCount > posCount)
                {
                    returnValue.prediction = false;
                    returnValue.predictionVal = -1;
                }
                else
                {
                    returnValue.predictionVal = 0;
                }

            }

            logger.LogInformation("Sentiment is {0} with a confidence of {1}.",returnValue.predictionVal,returnValue.predictionConfidence);
            
            return (returnValue);
        }

        private void splitAndCombine(ref List<string> tokenized, string text)
        {
            var tp = text.Split(" ");

            foreach (string s in tp)
            {
                tokenized.Add(s.ToLower());
            }

            for (int i = 0; i < tokenized.Count; i++)
            {
                if (tokenized[i] == "not")
                {
                    if (i + 1 >= tokenized.Count)
                    {
                        break;
                    }

                    if (tokenized[i + 1] == "a")
                    {
                        if (i + 2 >= tokenized.Count)
                        {
                            break;
                        }

                        string combined = tokenized[i] + "_" + tokenized[i + 2];
                        tokenized[i] = combined;
                        tokenized[i + 2] = " ";
                    }
                    else
                    {
                        if (i + 1 >= tokenized.Count)
                        {
                            break;
                        }

                        string combined = tokenized[i] + "_" + tokenized[i + 1];
                        tokenized[i] = combined;
                        tokenized[i + 1] = " ";
                    }
                }
            }

        }



    }
}
