from json import load, loads
from numpy import array
import numpy.typing as npt


class Data_From_JsonL:
    file_path: str
    number_of_lines: int
    array_of_images: npt.ArrayLike
    image_input_pairs: dict

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.number_of_lines = 0
        self.image_input_pairs = {}
        self.array_of_images = []
        with open(file_path, "r") as data:
            for line in data:
                self.number_of_lines += 1

            for i in range(self.number_of_lines):
                self.array_of_images.insert(i, 0)

            data.close()

        with open(file_path, "r") as data_again:
            for line_two in data_again:
                if line_two.strip():
                    line_value = loads(line_two)
                    self.array_of_images.append(line_value)
            data_again.close()

    def get_number_of_lines(self) -> int:
        return self.number_of_lines

    def get_image_input_pairs(self) -> dict:
        return self.image_input_pairs

    def get_array_of_images(self) -> npt.ArrayLike:
        return self.array_of_images


def main():
    data = Data_From_JsonL("steps.jsonl")
    print(data.get_array_of_images())


if __name__ == "__main__":
    main()
