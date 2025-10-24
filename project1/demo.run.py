# demo_run.py
# =====================================
# 一键运行整合版
# 功能：Dataset 读取 + transforms 演示 + TensorBoard 可视化
# =====================================

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# =============================
# 1. 自定义 Dataset 类
# =============================
class MyData(Dataset):  # 继承 Dataset
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 主目录
        self.label_dir = label_dir  # 类别子目录
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)  # 获取所有文件名

    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)  # 打开图片
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


# =============================
# 2. 数据路径初始化
# =============================
root_dir = "hymenoptera_data/hymenoptera_data/train"  # 你的原始路径
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

# 测试取出一张图
img, label = bees_dataset[1]
print(f"当前图片类别: {label}")
img.show()

# =============================
# 3. TensorBoard 基础初始化
# =============================
writer = SummaryWriter("Logs")

# =============================
# 4. transforms 演示
# =============================

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print("原始像素值：", img_tensor[0][0][0].item())
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print("归一化后像素值：", img_norm[0][0][0].item())
writer.add_image("Normalize", img_norm)

# Resize
print("原始尺寸：", img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)
print("调整后尺寸：", img_resize.size)

# Compose (Resize + ToTensor)
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_2", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(100, 50)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(5):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

# =============================
# 5. TensorBoard 标量曲线测试
# =============================
for i in range(100):
    writer.add_scalar("y=x", i, i)

# =============================
# 6. 图片写入演示
# =============================
img_path = "练手数据集/train/ants_image/5650366_e22b7e1065.jpg"
if os.path.exists(img_path):
    img_PIL = Image.open(img_path)
    img_array = np.array(img_PIL)
    writer.add_image("TestImage", img_array, 2, dataformats='HWC')
    print(f"已写入测试图片 {img_path}")
else:
    print("⚠️ 未找到练手数据集路径，可忽略此项。")

# =============================
# 7. 结束与提示
# =============================
writer.close()
print("所有任务完成！")
print("在终端输入tensorboard --logdir=Logs启动 TensorBoard")
print("tensorboard --logdir=Logs")
print("然后打开浏览器访问：http://localhost:6006")
