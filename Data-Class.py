import json
class Data_From_JsonL():
    file_path: str
    number_of_lines: int
    image_input_pairs: dict

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.number_of_lines = 0
        self.image_input_pairs = dict()
        with open(file_path) as data:
            for line in data:
               self.number_of_lines += 1

    def get_number_of_lines(self) -> int:
        return self.number_of_lines

    def get_image_input_pairs(self) -> dict:
        return self.image_input_pairs



def main():
    data = Data_From_JsonL("steps.jsonl")
    print(data.get_image_input_pairs().values())
    print(data.get_number_of_lines())


if __name__ == "__main__":
    main()
