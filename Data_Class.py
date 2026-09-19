from json import load, loads
from torch import tensor
from PIL import Image


class Data_From_JsonL:
    file_path: str
    number_of_lines: int
    list_of_dictionaries: list

    def __init__(self, file_path: str) -> None:
        """
        This is a special method which is used to initialize an object made from this class.

        @author Aaron Ilyas
        """

        self.file_path = file_path
        self.number_of_lines = 0
        self.list_of_dictionaries = []
        try:
            with open(file_path, "r") as data:
                for line in data:
                    self.number_of_lines += 1

            data.close()
        except Exception as e:
            file_path = ""
            # raise e

    def get_number_of_lines(self) -> int:
        """
        This method returns the number of lines from the jsonl file that was read in the __init__ method

        @author Aaron Ilyas
        """
        return self.number_of_lines

    def get_list_of_dictionaries(self) -> list:
        """
        This method returns a list consisting of all the dictionaries each of which contain the images name, starting frame, ending frame, and the expected inputs

        @author Aaron Ilyas
        """
        return self.list_of_dictionaries

    def get_tensor_of_gray_scale_pixel_values_and_its_dictionary(self, index) -> tuple:
        """
        This method returns both an tensor consisting of the gray scale pixel values for an image at the index you specify, as well as
        the dictionary that image is contained in.

        @author Aaron Ilyas
        """
        image_name = self.list_of_dictionaries[index]["image"]
        image_name = "first_gym_001/" + image_name
        img = Image.open(image_name)
        gray_img = img.convert("L")
        img_data = tensor(list(gray_img.get_flattened_data()))  # type: ignore
        return img_data, self.list_of_dictionaries[index]
