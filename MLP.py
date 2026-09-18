from torch import Tensor, argmax, dropout, optim, tensor
import torch
import torch.nn as nn
import Data_Class


class MLP(nn.Module):
    data_set: Data_Class.Data_From_JsonL
    input_value: Tensor
    number_of_activations_for_hidden_layer: int
    optimizer: optim.Adam
    model: nn.Sequential

    def __init__(self, data_set: Data_Class.Data_From_JsonL) -> None:
        super().__init__()
        self.data_set = data_set
        self.input_value = (
            self.data_set.get_tensor_of_gray_scale_pixel_values_and_its_dictionary(0)[0]
        )
        self.number_of_activations_for_input_layer = len(self.input_value)

        self.model = nn.Sequential(
            nn.Linear(self.number_of_activations_for_input_layer, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Softmax(),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def relate_input_with_number(self, input_button: str) -> int:
        if len(input_button) <= 0:
            return 9
        if input_button[0] == "UP":
            return 1
        if input_button[0] == "DOWN":
            return 2
        if input_button[0] == "LEFT":
            return 3
        if input_button[0] == "RIGHT":
            return 4
        if input_button[0] == "A":
            return 5
        if input_button[0] == "B":
            return 6
        if input_button[0] == "START":
            return 7
        if input_button[0] == "END":
            return 8
        else:
            return 9

    def find_input_from_number(self, input_number: int) -> str:
        if input_number == 1:
            return "UP"
        if input_number == 2:
            return "DOWN"
        if input_number == 3:
            return "LEFT"
        if input_number == 4:
            return "RIGHT"
        if input_number == 5:
            return "A"
        if input_number == 6:
            return "B"
        if input_number == 7:
            return "START"
        if input_number == 8:
            return "END"
        else:
            return "NONE"

    def train_network(self, number_of_epochs: int) -> None:
        criterion = nn.CrossEntropyLoss()

        for epoch in range(0, number_of_epochs, 37):
            pixels, label_info = (
                self.data_set.get_tensor_of_gray_scale_pixel_values_and_its_dictionary(
                    epoch
                )
            )
            input_data = pixels.float() / 255.0

            expected_outputs = self.relate_input_with_number(label_info["inputs"])
            target = torch.tensor([expected_outputs], dtype=torch.long)

            prediction = self.model(input_data)
            if prediction.ndim == 1:
                prediction = prediction.unsqueeze(0)

            self.optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            self.optimizer.step()

            """print(
                "Loss:",
                float(loss),
                "Prediction:",
                prediction.detach(),
                "Expected:",
                self.find_input_from_number(expected_outputs),
            )"""

    def load_models_weights_and_biases(self) -> None:
        self.model.load_state_dict(torch.load("models_weights_and_biases.pth"))

    def use_network(self, input_data: torch.Tensor) -> str:
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(torch.float32)
            models_output = max(self.model(input_data))
            return self.find_input_from_number(models_output)

    def save_weights_and_biases(self) -> None:
        torch.save(self.model.state_dict(), "models_weights_and_biases.pth")


def main():
    data = Data_Class.Data_From_JsonL("steps.jsonl")
    mlp_1 = MLP(data)
    # mlp_1.train_network(13542)
    # data = data.get_tensor_of_gray_scale_pixel_values_and_its_dictionary(0)[0]


if __name__ == "__main__":
    main()
