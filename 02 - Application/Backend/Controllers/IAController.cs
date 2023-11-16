using Microsoft.AspNetCore.Cors;
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("IA")]
[EnableCors("MainPolicy")]
public class NaiveBayesController : ControllerBase
{

    [HttpGet("naive-bayes/{message}")]
    public async Task<ActionResult<string>> NaiveBayesPredict(string message)
    {
        HttpClient client = new();
        string naiveBayesURI = "http://localhost:3030/API/naive-bayes/";

        try
        {
            string res = await client.GetStringAsync(naiveBayesURI + message);
            return Ok(res);
        }
        catch (System.Exception error)
        {
            return BadRequest(error.Message);
        }
    }

  
    [HttpGet("decision-tree/{message}")]
    public async Task<ActionResult<string>> DecisionTreePredict(string message)
    {
        HttpClient client = new();
        string decisionTreeURI = "http://localhost:3030/API/decision-tree/";

        try
        {
            string res = await client.GetStringAsync(decisionTreeURI + message);
            return Ok(res);
        }
        catch (System.Exception error)
        {
            return BadRequest(error.Message);
        }
    }
}