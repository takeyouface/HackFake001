import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from DFLIMG import *
from facelib import XSegNet


def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 没有找到. 请确保它存在.')

    if not model_path.exists():
        raise ValueError(f'{model_path} 没有找到. 请确保它存在.')
        
    io.log_info(f'应用训练好的XSEG遮罩到 {input_path.name} 文件夹下人脸素材中.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)
        
    xseg = XSegNet(name='XSeg', 
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    res = xseg.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "处理中"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath}不符合DFL使用图像标准！')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        if w != res:
            img = cv2.resize( img, (res,res), interpolation=cv2.INTER_CUBIC )        
            if len(img.shape) == 2:
                img = img[...,None]            
            
        mask = xseg.extract(img)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1
        
        dflimg.set_xseg_mask(mask)
        dflimg.save()


        
def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 没有找到. 请确保它存在.')
    
    output_path = input_path.parent / ('../EDIT')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'召回编辑XSEG遮罩的人脸素材到 {output_path.name}编辑目录！')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_copied = 0
    for filepath in io.progress_bar_generator(images_paths, "处理中"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath}不符合DFL使用图像标准！')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied += 1
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'拷贝文件数: {files_copied}')
    
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 没有找到. 请确保它存在.')
    
    io.log_info(f'正在处理文件夹 {input_path}')
    io.log_info('!!! 警告 : 应用的XSEG遮罩将从人脸素材中移除！！！')
    io.log_info('!!! 警告 : 应用的XSEG遮罩将从人脸素材中移除！！！')
    io.log_info('!!! 警告 : 应用的XSEG遮罩将从人脸素材中移除！！！')
    io.input_str('按回车键确认并继续.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "处理中"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath}不符合DFL使用图像标准！')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'已处理的文件: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 没有找到. 请确保它存在.')
    
    io.log_info(f'正在处理文件夹 {input_path}')
    io.log_info('!!! 警告 : 编辑好的XSeg遮罩样板将从人脸素材中移除 !!!')
    io.log_info('!!! 警告 : 编辑好的XSeg遮罩样板将从人脸素材中移除 !!!')
    io.log_info('!!! 警告 : 编辑好的XSeg遮罩样板将从人脸素材中移除 !!!')
    io.input_str('按回车键确认并继续.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "处理中"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath}不符合DFL使用图像标准！')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'已处理的文件: {files_processed}')
