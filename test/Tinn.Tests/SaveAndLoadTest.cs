using AutoFixture.Xunit2;
using FluentAssertions;
using Xunit;

namespace Tinn.Tests
{
    public class SaveAndLoadTest
    {
        private const string _filePath = "network.tinn";

        [Theory, AutoData]
        public void I_should_be_able_to_save_and_load_a_pretrained_neural_network(
            int inputCount, 
            int hiddenCount, 
            int outputCount)
        {
            // Arrange
            var originalNetwork = new TinyNeuralNetwork(inputCount, hiddenCount, outputCount);

            // Act
            originalNetwork.Save(_filePath);
            var loadedNetwork = TinyNeuralNetwork.Load(_filePath);

            // Assert
            loadedNetwork.Should().BeEquivalentTo(originalNetwork);
        }
    }
}
