class cityscapes_config(object):
    weights = [ 3.03871497 , 13.01999212, 4.53008444 , 38.00898985, 35.35990217,
                31.14247824, 45.82106668, 39.44451896, 6.00002668 , 32.64759009,
                17.34725897, 31.45715227, 47.0853521 , 11.70166965, 44.50966853,
                44.80501487, 45.63057065, 48.21080448, 41.21912833]
    ignore_index=250
    num_classes = 19
    color_classes = [
        ('road', (128, 64, 128), 0),
        ('sidewalk', (244, 35, 232), 1),
        ('building', (70, 70, 70), 2),
        ('wall', (102, 102, 156), 3),
        ('fence', (190, 153, 153), 4),
        ('pole', (153, 153, 153), 5),
        ('traffic_light', (250, 170, 30), 6),
        ('traffic_sign', (220, 220, 0), 7),
        ('vegetation', (107, 142, 35), 8),
        ('terrain', (152, 251, 152), 9),
        ('sky', (0, 130, 180), 10),
        ('person', (220, 20, 60), 11),
        ('rider', (255, 0, 0), 12),
        ('car', (0, 0, 142), 13),
        ('truck', (0, 0, 70), 14),
        ('bus', (0, 60, 100), 15),
        ('train', (0, 80, 100), 16),
        ('motorcycle', (0, 0, 230), 17),
        ('bicycle', (119, 11, 32), 18),
        ('void', (0, 0, 0), ignore_index)
    ]  # 19 classes + 1 void class
    swnet_weight_path = './weights/cityscapes-swnet-R18.pt'
    bsnet_weight_path = './weights/cityscapes-bisenet-R18.pth'
    #resume_path = './weights/gsvnet_bisenet_r18.tar'
    resume_path = './weights/gsvnet_swnet_r18.tar'
    optical_flow_network_path = './weights/flownet.pth.tar'
    data_path = '' # put your data path here