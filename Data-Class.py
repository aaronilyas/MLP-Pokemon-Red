from json import load, loads
from torch import tensor
from PIL import Image


class Data_From_JsonL:
    file_path: str
    number_of_lines: int
    list_of_dictionaries: list

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.number_of_lines = 0
        self.list_of_dictionaries = []
        with open(file_path, "r") as data:
            for line in data:
                self.number_of_lines += 1

            data.close()

        with open(file_path, "r") as data_again:
            for line_two in data_again:
                if line_two.strip():
                    line_value = line_two
                    line_value = dict(loads(line_value))
                    self.list_of_dictionaries.append(line_value)
            data_again.close()

    def get_number_of_lines(self) -> int:
        return self.number_of_lines

    def get_list_of_dictionaries(self) -> list:
        return self.list_of_dictionaries

    def get_tensor_of_gray_scale_pixel_values_from_list(self, index):
        image_name = self.list_of_dictionaries[index]["image"]
        image_name = "first_gym_001/" + image_name
        img = Image.open(image_name)
        gray_img = img.convert("L")
        img_data = tensor(list(gray_img.get_flattened_data()))  # type: ignore
        return img_data


def main():
    data = Data_From_JsonL("steps.jsonl")


if __name__ == "__main__":
    main()
