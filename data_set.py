from torch.utils.data import Dataset
from PIL import Image
import torch

def read_file(file_path):
    '''
    Input: Path of the text file.
    Output: Returns the list of list. Each list consist of 6 elements, where first five are images and last is label. 
    '''
    final_list = []
    with open(file_path, 'r') as f_p:
        while (1):
            # reading individual line from the text file
            current_line = f_p.readline()
            if not current_line:
                break               # break if the line is empty. (i.e. reached at the end of the file) 
            
            temp_list = current_line.strip().split()   # strip(removes space in the begining and end of line)
                                                       # split(convert string into list, seperator is whitespace) 

            final_list.append(temp_list)
    
    return final_list

class tvtDatasetList(Dataset):
    '''
    torch.utils.data.Dataset is an abstract class representing a dataset. 
    tvtDatasetList inherit Dataset and override __len__ and __getitem__methods.
    Reference - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    def __init__(self, file_path, transforms):
        self.list_of_images = read_file(file_path)        # list of lists.   
        self.transforms = transforms                     # Tranform the images to the specified tensor format.

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        image_list = self.list_of_images[idx]            # will load one particular list. (In our case, there are 5 image paths and 1 ground truth)
        temp = []
        for i in range(len(image_list)-1):
            image = Image.open(image_list[i])                       # reading the image
            transform_image = self.transforms(image)                # Applying transformation
            temp.append(torch.unsqueeze(transform_image, dim=0))    

        data = torch.cat(temp, 0)
        label = Image.open(image_list[len(image_list)-1])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

