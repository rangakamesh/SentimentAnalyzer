using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using OpenCognition.Models.SentimentPack;

namespace OpenCognition.PredictSentiment
{
    public static class PredictSentiment
    {
        [FunctionName("PredictSentiment")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("Sentiment analyzer processed a request.");

            string text = req.Query["text"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            text = text ?? data?.text;

            if(text==null)
            {
                return new BadRequestObjectResult("Input parameter \"text\" is required. ex : text=is this a bad sentence?");
            }
            else
            {
                SentimentClassifier sc = new SentimentClassifier(log);
                var pred = sc.predict(text);
                var returnValue = new SentimentReturn(pred.predictionVal, pred.predictionConfidence);
                return new OkObjectResult(returnValue);
            }

        }

        public class SentimentReturn
        {
            public int prediction { get; set; }
            public float confidence { get; set; }

            public SentimentReturn(int _prediction, float _confidence)
            {
                prediction = _prediction;
                confidence = _confidence;
            }
        }
    }
}
