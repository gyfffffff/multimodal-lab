import logging
import os
import time
class logger:
    def __init__(self, log_dir, version):
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            filename=os.path.join(log_dir, version+'.txt'),
            filemode='a+',
            level=logging.INFO)
        
    def write(self, info):
        print(info)
        logging.info(info)

    def write_config(self, args):
        for k, v in vars(args).items():
            logging.info(k+'\t'+str(v))