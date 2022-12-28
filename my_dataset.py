import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DriveDataset(Dataset): 
    def __init__(self, root, is_train, transform=None): 
        # type: (str, bool, None) -> None 
        super(DriveDataset, self).__init__() 

        self.flag = 'train' if is_train else 'test' 
        data_root = os.path.join(root, 'DRIVE', self.flag) 
        assert os.path.exists(data_root), "数据路径不存在" 
        self.transform = transform 
        img_names = [i for i in os.listdir(os.path.join(data_root, 'image')) 
                     if i.endswith('.tif')] 
        self.img_paths = [os.path.join(data_root, 'images', i) for i in img_names] 
        self.manual = [os.path.join(data_root, '1st_manual', i.split('_')[0] + 
                       '_manual1.gif') for i in img_names] 
        
        # 检查文件是否存在 
        for i in self.manual: 
            if not os.path.exists(i): 
                raise FileNotFoundError("文件不存在") 

        self.roi_mask = [os.path.join(data_root, 'mask', i.split('_')[0] + 
                         f"_{self.flag}_mask.gif") for i in img_names] 

        for i in self.roi_mask: 
            if not os.path.exists(i): 
                raise FileNotFoundError("文件不存在") 

    def __len__(self): 
        return len(self.img_paths) 

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB') 
        manual = Image.open(self.manual[index]).convert('L') 
        # 将前景转化为1 背景转化为0 
        manual = np.array(manual) / 255.0 
        roi_mask = Image.open(self.roi_mask[index]).convert('L') 
        # 将不进行计算的地方转化为255 
        roi_mask = 255 - np.array(roi_mask) 
        # 防止有极个别的像素相加后越界 
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)  

        mask = Image.fromarray(mask) 

        if self.transform is not None: 
            img, mask = self.transform(img, mask) 

        return img, mask 

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


