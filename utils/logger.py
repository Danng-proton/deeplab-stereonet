import logging
import os


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )#这个类配置了日志的格式，在里面自定义设置日期和时间，输出日志的时候将会按照设置的格式显示内容。

    #下：创建一个logger
    logger = logging.getLogger('example')#logging.getLogger(name)获取logger对象，如果不指定name则返回root对象
    handler = logging.StreamHandler()#创建一个handler，用于输出到控制台
    handler.setFormatter(file_formatter)#设置日志的格式
    logger.addHandler(handler)#将logger添加到handler里面

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger