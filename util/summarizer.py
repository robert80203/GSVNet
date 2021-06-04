import os
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from torchvision.transforms import transforms as standard_transforms

class TensorboardSummary(object):
    def __init__(self, directory, classes):
        self.directory = directory
        self.writer = self.create_summary()
        self.classes = classes
        self.to_tensor = standard_transforms.ToTensor()

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def log_text(self, tag, text, epoch):
        self.writer.add_text(tag, text, epoch)

    def log_image(self, title, image, global_step ):
        image = image.permute(1, 2, 0).numpy()
        image = np.uint8(image)
        self.writer.add_image(title, self.to_tensor(Image.fromarray(image)), global_step)

    def log_train_loss(self, train_loss, global_step):
        self.writer.add_scalar('loss/train', train_loss, global_step)

    def log_val_loss(self, val_loss, global_step):
        self.writer.add_scalar('loss/val', val_loss, global_step)

    def log_loss(self, tag, loss, global_step):
        self.writer.add_scalar('loss/' + str(tag), loss, global_step)

    def log_evaluation(self, prefix, pa, pac, mIoU, fwIoU, global_step):
        self.writer.add_scalar(prefix + "/mIoU", mIoU, global_step)
        self.writer.add_scalar(prefix + "/fwIoU", fwIoU, global_step)
        self.writer.add_scalar(prefix + "/pixel_acc", pa, global_step)
        self.writer.add_scalar(prefix + "/pixel_acc_class", pac, global_step)

    def log_cm_evaluation(self, prefix, auroc, pr, pr2, acc, f1, global_step):
        self.writer.add_scalar(prefix + "/auroc", auroc, global_step)
        self.writer.add_scalar(prefix + "/aupr1", pr, global_step)
        self.writer.add_scalar(prefix + "/aupr0", pr2, global_step)
        self.writer.add_scalar(prefix + "/acc", acc, global_step)
        self.writer.add_scalar(prefix + "/mIoU", f1, global_step)
