using System.Globalization;
using ShellProgressBar;
using Tinn;

const string datasetUri = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data";
const string dataSetFileName = "semeion.data";
const int inputCount = 256;
const int hiddenCount = 28;
const int outputCount = 10;

// reserve ~10% of data for verification
const int verifyCount = 150;

const int learningIterations = 10;
var learningRate = 0.1f;
const float learningRateDecay = 0.95f;
var random = new Random(0);

if (File.Exists(dataSetFileName) == false)
{
    Console.WriteLine("Downloading MNIST dataset...");
    using HttpClient client = new();
    using Stream file = File.Create(dataSetFileName);
    var stream = await client.GetStreamAsync(datasetUri);
    stream.CopyTo(file);
    Console.WriteLine("Download completed.");
}

var allData = File.ReadAllLines(dataSetFileName)
    .Select(line => line.Split(" ").Select(x => float.Parse(x, CultureInfo.InvariantCulture)))
    .Select(number => new DataItem(
        Input: number.Take(inputCount).ToArray(),
        Output: number.Skip(inputCount).Take(outputCount).ToArray())
    )
    .ToList();

var learningData = allData.Skip(verifyCount).ToList();
var verifyData = allData.Take(verifyCount).ToList();

var network = new TinyNeuralNetwork(inputCount, hiddenCount, outputCount);
var progress = new ProgressBar(learningIterations, "Training...");
var currentAccuracy = 0.0;
for (var i = 0; i < learningIterations; i++)
{
    using var child = progress.Spawn(
        learningData.Count,
        $"Iteration {i}",
        new ProgressBarOptions { CollapseWhenFinished = true });
    foreach (var (input, output) in learningData)
    {
        network.Train(input, output, learningRate);
        child.Tick();
    }

    Shuffle(learningData);
    learningRate *= learningRateDecay;

    currentAccuracy = ComputeAccuracy(verifyData, network);
    progress.Tick($"Achieved {currentAccuracy:P2} accuracy.");
    await Task.Delay(TimeSpan.FromSeconds(0.5));
}

network.Save("network.tinn");
currentAccuracy = ComputeAccuracy(verifyData, network);
Console.WriteLine($"Achieved {currentAccuracy:P2} accuracy.");

void Shuffle<T>(List<T> list)
{
    for (var i = 0; i < list.Count; i++)
    {
        var j = random.Next(list.Count);
        (list[i], list[j]) = (list[j], list[i]);
    }
}

double ComputeAccuracy(IEnumerable<DataItem> subset, TinyNeuralNetwork network)
{
    var predictedNumbers = subset
        .Select(x => network.Predict(x.Input))
        .Select(f => f.Select((n, i) => (n, i)).Max().i)
        .ToArray();

    var actualNumbers = subset
        .Select(record => record.Output.Select((n, i) => (n, i)).Max().i)
        .ToArray();

    var correctlyGuessed = predictedNumbers.Zip(actualNumbers, (l, r) => l == r ? 1.0 : 0.0).Sum();
    var accuracy = correctlyGuessed / actualNumbers.Length;
    return accuracy;
}

record struct DataItem(float[] Input, float[] Output);
