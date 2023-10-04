using FluentAssertions;

namespace Tinn.Tests;

public class SaveAndLoadTest
{
    private const string FilePath = "network.tinn";

    [Theory, AutoData]
    public void I_should_be_able_to_save_and_load_a_pre_trained_neural_network(
        int inputCount,
        int hiddenCount,
        int outputCount)
    {
        // Arrange
        var originalNetwork = new TinyNeuralNetwork(inputCount, hiddenCount, outputCount);

        // Act
        originalNetwork.Save(FilePath);
        var loadedNetwork = TinyNeuralNetwork.Load(FilePath);

        // Assert
        loadedNetwork.Should().BeEquivalentTo(originalNetwork);
    }

    [Theory, AutoData]
    public void I_should_be_able_to_construct_from_existing_weights(
        int inputCount,
        int hiddenCount,
        int outputCount)
    {
        // Arrange
        var originalNetwork = new TinyNeuralNetwork(inputCount, hiddenCount, outputCount);

        // Act
        var newNetwork = new TinyNeuralNetwork(
            originalNetwork.Weights,
            originalNetwork.Biases,
            inputCount,
            hiddenCount,
            outputCount);

        // Assert
        newNetwork.Should().BeEquivalentTo(originalNetwork);
    }
}
