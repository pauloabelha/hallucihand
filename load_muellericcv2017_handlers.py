import importlib.machinery

def load():
    loader_io_image = importlib.machinery.SourceFileLoader('io_image', '/home/paulo/muellerICCV2017/io_image.py')
    io_image = loader_io_image.load_module('io_image')
    loader_converter = importlib.machinery.SourceFileLoader('converter', '/home/paulo/muellerICCV2017/converter.py')
    converter = loader_converter.load_module('converter')
    loader_camera = importlib.machinery.SourceFileLoader('camera', '/home/paulo/muellerICCV2017/camera.py')
    camera = loader_camera.load_module('camera')
    loader_dataset_handler = importlib.machinery.SourceFileLoader('dataset_handler', '/home/paulo/muellerICCV2017/dataset_handler.py')
    dataset_handler = loader_dataset_handler.load_module('dataset_handler')
    loader_ergodexter_handler = importlib.machinery.SourceFileLoader('egodexter_handler', '/home/paulo/muellerICCV2017/egodexter_handler.py')
    egodexter_handler = loader_ergodexter_handler.load_module('egodexter_handler')
    loader_synthhands_handler = importlib.machinery.SourceFileLoader('synthhands_handler', '/home/paulo/muellerICCV2017/synthhands_handler.py')
    synthhands_handler = loader_synthhands_handler.load_module('synthhands_handler')
    return synthhands_handler, egodexter_handler