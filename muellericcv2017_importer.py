import os
import importlib.machinery

def load_modules(project_name='muellericcv2017'):
    module_names = ['io_image', 'converter', 'camera', 'dataset_handler',
                    'egodexter_handler', 'synthhands_handler', ]
    modules = {}
    for module_name in module_names:
        abs_path_module = os.path.abspath(os.path.join(__file__, '..', '..')) +\
                          '/' + project_name + '/' + module_name + '.py'
        loader = importlib.machinery.SourceFileLoader(module_name, abs_path_module)
        module = loader.load_module(module_name)
        modules[module_name] = module
    return modules

def load():
    loader_io_image = importlib.machinery.SourceFileLoader('io_image', '../muellerICCV2017/io_image.py')
    io_image = loader_io_image.load_module('io_image')
    loader_converter = importlib.machinery.SourceFileLoader('converter', '../muellerICCV2017/converter.py')
    converter = loader_converter.load_module('converter')
    loader_camera = importlib.machinery.SourceFileLoader('camera', '../muellerICCV2017/camera.py')
    camera = loader_camera.load_module('camera')
    loader_dataset_handler = importlib.machinery.SourceFileLoader('dataset_handler', '../muellerICCV2017/dataset_handler.py')
    dataset_handler = loader_dataset_handler.load_module('dataset_handler')
    loader_ergodexter_handler = importlib.machinery.SourceFileLoader('egodexter_handler', '../muellerICCV2017/egodexter_handler.py')
    egodexter_handler = loader_ergodexter_handler.load_module('egodexter_handler')
    loader_synthhands_handler = importlib.machinery.SourceFileLoader('synthhands_handler', '../muellerICCV2017/synthhands_handler.py')
    synthhands_handler = loader_synthhands_handler.load_module('synthhands_handler')
    return synthhands_handler, egodexter_handler