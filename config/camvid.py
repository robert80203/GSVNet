class camvid_config(object):
    weights = [ 6.088367253290739, 4.519555764402027, 33.97442457184865,
                3.892793490946617, 12.472488354029295, 8.53161322681152,
                47.70112012347462, 29.66528556102586, 18.998198498821438,
                38.38783950878698, 40.105909294306784]
    ignore_index = 250
    num_classes = 11
    color_classes = [
        ('Sky', (128, 128, 128), 0),  # 0
        ('Building', (128, 0, 0), 1),  # 1
        ('Column-Pole', (192, 192, 128), 2),  # 2
        ('Road', (128, 64, 128), 3),  # 3
        ('Sidewalk', (0, 0, 192), 4),  # 4
        ('Tree', (128, 128, 0), 5),  # 5
        ('Sign-Symbol', (192, 128, 128), 6),  # 6
        ('Fence', (64, 64, 128), 7),  # 7
        ('Car', (64, 0, 128), 8),  # 8
        ('Pedestrain', (64, 64, 0), 9),  # 9
        ('Bicyclist', (0, 128, 192), 10),  # 10
        ('Void', (0, 0, 0), ignore_index),  # 11
    ]
    swnet_weight_path = './weights/cityscapes-swnet-R18.pt'
    bsnet_weight_path = './weights/cityscapes-bisenet-R18.pth'


    #resume_path = './weights/gsvnet_bisenet_r18.tar'
    resume_path = './weights/gsvnet_swnet_r18.tar'

    optical_flow_network_path = './weights/flownet.pth.tar'
    data_path = '' #put your data path here