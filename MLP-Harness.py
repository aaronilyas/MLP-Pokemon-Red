import PIL as pil
from Data_Class import Data_From_JsonL
from MLP import MLP
import pyboy as gb
import torch
import numpy as np


def main():
    mlp = MLP(Data_From_JsonL(""), False)
    rom = gb.PyBoy("Pokemon - Red Version (USA, Europe).gb")
    with rom as game:
        game.tick()
        for i in range(13500):
            screen_image = game.screen.image
            if screen_image is not None:
                gray_scale_values_of_image = screen_image.convert("L")
                pixels = (
                    torch.tensor(
                        gray_scale_values_of_image.getdata(), dtype=torch.float32
                    ).reshape(1, -1)
                    / 255.0
                )
                action = mlp.use_network(pixels)
                if action != "NONE":
                    game.button(action)
                    game.tick()
                    print(action)
                else:
                    game.tick()
                    print(action)


if __name__ == "__main__":
    main()
