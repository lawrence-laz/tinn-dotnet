namespace Tinn.Tests;

[UsesVerify]
public class LoadTest
{
    [Theory, InlineAutoData("2021-01-18.tinn")]
    public async Task I_should_be_able_to_load_a_trained_neural_network_from_2021(string filePath)
    {
        // Act
        var actual = TinyNeuralNetwork.Load(filePath);

        // Assert
        await Verify(new
        {
            InputCount = actual.InputCount,
            Hidden = actual.HiddenLayer,
            Output = actual.OutputLayer,
            Weights = actual.Weights,
            Biases = actual.Biases,
        });
    }
}
