import codecs
import os
import random
import shutil
from PIL import Image

train_ratio = 4.0 / 5
#需要修改的数据集路径
all_file_dir = 'PaddleClas/dataset/NEU-CLS'
class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c)) and not c.endswith('Set') and not c.startswith('.')]
class_list.sort()
print(class_list)
train_image_dir = os.path.join(all_file_dir, "trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
    
eval_image_dir = os.path.join(all_file_dir, "evalImageSet")
if not os.path.exists(eval_image_dir):
    os.makedirs(eval_image_dir)

train_file = codecs.open(os.path.join(all_file_dir, "train.txt"), 'w')
eval_file = codecs.open(os.path.join(all_file_dir, "eval.txt"), 'w')

with codecs.open(os.path.join(all_file_dir, "label_list.txt"), "w") as label_list:
    label_id = 0
    for class_dir in class_list:
        label_list.write("{0}\t{1}\n".format(label_id, class_dir))
        image_path_pre = os.path.join(all_file_dir, class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre, file))
                if random.uniform(0, 1) <= train_ratio:
                    shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(train_image_dir, file))
                    train_file.write("{0} {1}\n".format(os.path.join("trainImageSet", file), label_id))
                else:
                    shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(eval_image_dir, file))
                    eval_file.write("{0} {1}\n".format(os.path.join("evalImageSet", file), label_id))
            except Exception as e:
                pass
                # 存在一些文件打不开，此处需要稍作清洗
        label_id += 1        
train_file.close()
eval_file.close()