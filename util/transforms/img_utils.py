import cv2
import numpy as np
import numbers
import random
import collections
import torch
import torch
import torch.nn.functional as F
from PIL import Image


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))

def generate_transform_point(scale_factor_low, scale_factor_high, height, width, crop_height, crop_width, do_horizontal_flip):
    # scale first
    #scale_factor = np.random.randint(scale_factor_low * 10, scale_factor_high * 10 + 1) / 10
    scale_weights = [0.75,1.0,1.5,1.75,2.0]
    scale_factor = scale_weights[np.random.randint(0,len(scale_weights)-1)]
    scaled_height, scaled_width = int(height * scale_factor),  int(width * scale_factor)

    # random horizontal flip
    do_horizontal_flip = False
    if np.random.randint(10) / 10 > 0.5:
        do_horizontal_flip = True

    if scaled_height < crop_height or scaled_width < crop_width:
        return {
            'scale': scale_factor,
            'scaled_point': (0, 0),
            'rescale_size': (crop_height, crop_width),
            'rescale': True,
            'hor_flip': do_horizontal_flip
        }
    else:
        y_point = np.random.randint(low=0, high=scaled_height - crop_height + 1)
        x_point = np.random.randint(low=0, high=scaled_width - crop_width + 1)
        return {
            'scale': scale_factor,
            'scaled_point': (y_point, x_point),
            'rescale_size': (crop_height, crop_width),
            'rescale': False,
            'hor_flip': do_horizontal_flip
        }


def transform_map(feature_map , transform_point, interpolation_mode="bilinear", use_crop=True):
    if feature_map is None: return feature_map

    assert len(feature_map.size()) == 4 or len(feature_map.size()) == 5


    back_to_five = False
    if len(feature_map.size()) == 5:
        batch_size, feat_map = feature_map.size()[:2]
        feature_map = feature_map.view(batch_size * feat_map, *feature_map.size()[-3:])
        back_to_five = True

    scale_factor = transform_point['scale']
    scaled_point_y , scaled_point_x = transform_point['scaled_point']
    crop_height, crop_width = transform_point['rescale_size']

    is_align=None
    if interpolation_mode == "bilinear": is_align = True


    ### size of output should be devide by 32
    height, width = feature_map.size()[-2:]
    height, width = int(scale_factor * height), int(scale_factor * width)
    height = height //32 * 32
    width = width //32 * 32
    ### size of output should be devide by 32

    # Scale 0.5 - 2
    feature_map = F.interpolate(feature_map, size=(height, width), mode=interpolation_mode, align_corners=is_align)

    # Crop
    if use_crop:
        # Rescale if shorter
        if transform_point['rescale']:
            feature_map = F.interpolate(feature_map, size=transform_point['rescale_size'], mode=interpolation_mode, align_corners=is_align)

        feature_map = feature_map[:, :,
                      scaled_point_y: scaled_point_y + crop_height,
                      scaled_point_x: scaled_point_x + crop_width]

    if transform_point['hor_flip']:
        feature_map = torch.flip( feature_map,[3])

    if back_to_five:
        feature_map = feature_map.view(batch_size, feat_map, *feature_map.size()[-3:])
    return feature_map

def tensor_flip_channel(image):
    flipped_image = torch.cat([image[:,2].unsqueeze(1),image[:,1].unsqueeze(1),image[:,0].unsqueeze(1)],dim = 1)
    return flipped_image

'''
class RemapLabels:
    def __init__(self, mapping: dict, ignore_id, total=35):
        self.mapping = np.ones((max(total, max(mapping.keys())) + 1,), dtype=np.uint8) * ignore_id
        self.ignore_id = ignore_id
        for i in range(len(self.mapping)):
            self.mapping[i] = mapping[i] if i in mapping else ignore_id

    def _trans(self, labels):
        max_k = self.mapping.shape[0] - 1
        labels[labels > max_k] //= 1000
        labels = self.mapping[labels].astype(labels.dtype)
        return labels

    def __call__(self, example):
        if not isinstance(example, dict):
            return self._trans(example)
        if 'labels' not in example:
            return example
        ret_dict = {'labels': pimg.fromarray(self._trans(np.array(example['labels'])))}
        if 'original_labels' in example:
            ret_dict['original_labels'] = pimg.fromarray(self._trans(np.array(example['original_labels'])))
        return {**example, **ret_dict}




def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_mat(pic, mode=None):
    """Convert a tensor to an ndarray.
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
            mode (str): color space and pixel depth of input data (optional)
                        for example: cv2.COLOR_RGB2BGR.
        Returns:
            np.array: Image converted to PIL Image.
        """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.squeeze(np.transpose(pic.numpy(), (1, 2, 0)))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))
    if mode is None:
        return npimg

    else:
        return cv2.cvtColor(npimg, mode)


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin


def pad_image_size_to_multiples_of(img, multiple, pad_value):
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)


def resize_ensure_shortest_edge(img, edge_length,
                                interpolation_mode=cv2.INTER_LINEAR):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)

    return img


def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale


def random_scale_with_length(img, gt, length):
    size = random.choice(length)
    sh = size
    sw = size
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, size


def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,


def random_rotation(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt


def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img


def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def random_crop(img, gt, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions 3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy
'''