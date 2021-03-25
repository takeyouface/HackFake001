import numpy as np
import copy

from facelib import FaceType
from core.interact import interact as io


class MergerConfig(object):
    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    ####

    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0,
                       sharpen_mode=0,
                       blursharpen_amount=0,
                       **kwargs
                       ):
        self.type = type

        self.sharpen_dict = {0:"None", 1:'box', 2:'gaussian'}

        #default changeable params
        self.sharpen_mode = sharpen_mode
        self.blursharpen_amount = blursharpen_amount

    def copy(self):
        return copy.copy(self)

    #overridable
    def ask_settings(self):
        s = """选择锐化模式: \n"""
        for key in self.sharpen_dict.keys():
            s += f"""({key}) {self.sharpen_dict[key]}\n"""
        io.log_info(s)
        self.sharpen_mode = io.input_int ("", 0, valid_list=self.sharpen_dict.keys(), help_message="通过应用锐化模式增强细节。")

        if self.sharpen_mode != 0:
            self.blursharpen_amount = np.clip ( io.input_int ("选择模糊/模糊数值", 0, add_info="-100..100"), -100, 100 )

    def toggle_sharpen_mode(self):
        a = list( self.sharpen_dict.keys() )
        self.sharpen_mode = a[ (a.index(self.sharpen_mode)+1) % len(a) ]

    def add_blursharpen_amount(self, diff):
        self.blursharpen_amount = np.clip ( self.blursharpen_amount+diff, -100, 100)

    #overridable
    def get_config(self):
        d = self.__dict__.copy()
        d.pop('type')
        return d

    #overridable
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfig):
            return self.sharpen_mode == other.sharpen_mode and \
                   self.blursharpen_amount == other.blursharpen_amount

        return False

    #overridable
    def to_string(self, filename):
        r = ""
        r += f"锐化模式（N） : {self.sharpen_dict[self.sharpen_mode]}\n"
        r += f"锐化/模糊（Y/H） : {self.blursharpen_amount}\n"
        return r

mode_dict = {0:'original',
             1:'overlay',
             2:'hist-match',
             3:'seamless',
             4:'seamless-hist-match',
             5:'raw-rgb',
             6:'raw-predict'}

mode_str_dict = { mode_dict[key] : key for key in mode_dict.keys() }

mask_mode_dict = {1:'dst',
                  2:'learned-prd',
                  3:'learned-dst',
                  4:'learned-prd*learned-dst',
                  5:'learned-prd+learned-dst',
                  6:'FAN-prd',
                  7:'FAN-dst',
                  8:'FAN-prd*FAN-dst',
                  9:'learned*FAN-prd*FAN-dst',
                  11:'XSeg-prd',
                  12:'XSeg-dst',
                  13:'XSeg-prd*XSeg-dst',
                  14:'learned-prd*learned-dst*XSeg-prd*XSeg-dst'
                  }


ctm_dict = { 0: "None", 1:"rct", 2:"lct", 3:"mkl", 4:"mkl-m", 5:"idt", 6:"idt-m", 7:"sot-m", 8:"mix-m" }
ctm_str_dict = {None:0, "rct":1, "lct":2, "mkl":3, "mkl-m":4, "idt":5, "idt-m":6, "sot-m":7, "mix-m":8 }

class MergerConfigMasked(MergerConfig):

    def __init__(self, face_type=FaceType.FULL,
                       default_mode = 'overlay',
                       mode='overlay',
                       masked_hist_match=True,
                       hist_match_threshold = 238,
                       mask_mode = 4,
                       erode_mask_modifier = 0,
                       blur_mask_modifier = 0,
                       motion_blur_power = 0,
                       output_face_scale = 0,
                       super_resolution_power = 0,
                       color_transfer_mode = ctm_str_dict['rct'],
                       image_denoise_power = 0,
                       bicubic_degrade_power = 0,
                       color_degrade_power = 0,
                       **kwargs
                       ):

        super().__init__(type=MergerConfig.TYPE_MASKED, **kwargs)

        self.face_type = face_type
        if self.face_type not in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL, FaceType.WHOLE_FACE, FaceType.HEAD ]:
            raise ValueError("MergerConfigMasked不支持此类型的面部。")

        self.default_mode = default_mode

        #default changeable params
        if mode not in mode_str_dict:
            mode = mode_dict[1]

        self.mode = mode
        self.masked_hist_match = masked_hist_match
        self.hist_match_threshold = hist_match_threshold
        self.mask_mode = mask_mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        self.motion_blur_power = motion_blur_power
        self.output_face_scale = output_face_scale
        self.super_resolution_power = super_resolution_power
        self.color_transfer_mode = color_transfer_mode
        self.image_denoise_power = image_denoise_power
        self.bicubic_degrade_power = bicubic_degrade_power
        self.color_degrade_power = color_degrade_power

    def copy(self):
        return copy.copy(self)

    def set_mode (self, mode):
        self.mode = mode_dict.get (mode, self.default_mode)

    def toggle_masked_hist_match(self):
        if self.mode == 'hist-match':
            self.masked_hist_match = not self.masked_hist_match

    def add_hist_match_threshold(self, diff):
        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            self.hist_match_threshold = np.clip ( self.hist_match_threshold+diff , 0, 255)

    def toggle_mask_mode(self):
        a = list( mask_mode_dict.keys() )
        self.mask_mode = a[ (a.index(self.mask_mode)+1) % len(a) ]

    def add_erode_mask_modifier(self, diff):
        self.erode_mask_modifier = np.clip ( self.erode_mask_modifier+diff , -400, 400)

    def add_blur_mask_modifier(self, diff):
        self.blur_mask_modifier = np.clip ( self.blur_mask_modifier+diff , 0, 400)

    def add_motion_blur_power(self, diff):
        self.motion_blur_power = np.clip ( self.motion_blur_power+diff, 0, 100)

    def add_output_face_scale(self, diff):
        self.output_face_scale = np.clip ( self.output_face_scale+diff , -50, 50)

    def toggle_color_transfer_mode(self):
        self.color_transfer_mode = (self.color_transfer_mode+1) % ( max(ctm_dict.keys())+1 )

    def add_super_resolution_power(self, diff):
        self.super_resolution_power = np.clip ( self.super_resolution_power+diff , 0, 100)

    def add_color_degrade_power(self, diff):
        self.color_degrade_power = np.clip ( self.color_degrade_power+diff , 0, 100)

    def add_image_denoise_power(self, diff):
        self.image_denoise_power = np.clip ( self.image_denoise_power+diff, 0, 500)

    def add_bicubic_degrade_power(self, diff):
        self.bicubic_degrade_power = np.clip ( self.bicubic_degrade_power+diff, 0, 100)

    def ask_settings(self):
        s = """合成模式（数字1-6）: \n"""
        for key in mode_dict.keys():
            s += f"""({key}) {mode_dict[key]}\n"""
        io.log_info(s)
        mode = io.input_int ("", mode_str_dict.get(self.default_mode, 1) )

        self.mode = mode_dict.get (mode, self.default_mode )

        if 'raw' not in self.mode:
            if self.mode == 'hist-match':
                self.masked_hist_match = io.input_bool("使用直方图模式合并？", True)

            if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("设置匹配亮度数值。", 255, add_info="0..255"), 0, 255)

        s = """选择遮罩模式: \n"""
        for key in mask_mode_dict.keys():
            s += f"""({key}) {mask_mode_dict[key]}\n"""
        io.log_info(s)
        self.mask_mode = io.input_int ("", 1, valid_list=mask_mode_dict.keys() )

        if 'raw' not in self.mode:
            self.erode_mask_modifier = np.clip ( io.input_int ("设置遮罩侵蚀度,推荐数值25-75", 0, add_info="-400..400"), -400, 400)
            self.blur_mask_modifier =  np.clip ( io.input_int ("设置遮罩边缘模糊度,推荐数值50-120", 0, add_info="0..400"), 0, 400)
            self.motion_blur_power = np.clip ( io.input_int ("设置运动模糊度,不推荐设置", 0, add_info="0..100"), 0, 100)

        self.output_face_scale = np.clip (io.input_int ("设置合成输出面部大小,不推荐设置", 0, add_info="-50..50" ), -50, 50)

        if 'raw' not in self.mode:
            self.color_transfer_mode = io.input_str ( "是否应用色彩转换模式融合肤色，选择模式", None, valid_list=list(ctm_str_dict.keys())[1:] )
            self.color_transfer_mode = ctm_str_dict[self.color_transfer_mode]

        super().ask_settings()

        self.super_resolution_power = np.clip ( io.input_int ("设置超分辨率强度以增强清晰度", 0, add_info="0..100", help_message="Enhance details by applying superresolution network."), 0, 100)

        if 'raw' not in self.mode:
            self.image_denoise_power = np.clip ( io.input_int ("设置降噪数值", 0, add_info="0..500"), 0, 500)
            self.bicubic_degrade_power = np.clip ( io.input_int ("设置插值数值", 0, add_info="0..100"), 0, 100)
            self.color_degrade_power = np.clip (  io.input_int ("设置降质数值", 0, add_info="0..100"), 0, 100)

        io.log_info ("")

    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfigMasked):
            return super().__eq__(other) and \
                   self.mode == other.mode and \
                   self.masked_hist_match == other.masked_hist_match and \
                   self.hist_match_threshold == other.hist_match_threshold and \
                   self.mask_mode == other.mask_mode and \
                   self.erode_mask_modifier == other.erode_mask_modifier and \
                   self.blur_mask_modifier == other.blur_mask_modifier and \
                   self.motion_blur_power == other.motion_blur_power and \
                   self.output_face_scale == other.output_face_scale and \
                   self.color_transfer_mode == other.color_transfer_mode and \
                   self.super_resolution_power == other.super_resolution_power and \
                   self.image_denoise_power == other.image_denoise_power and \
                   self.bicubic_degrade_power == other.bicubic_degrade_power and \
                   self.color_degrade_power == other.color_degrade_power

        return False

    def to_string(self, filename):
        r = (
            f"""合成参数 {filename}:\n"""
            f"""模式（1-6）: {self.mode}\n"""
            )

        if self.mode == 'hist-match':
            r += f"""直方图遮罩（Z）: {self.masked_hist_match}\n"""

        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            r += f"""匹配亮度（Q/A）: {self.hist_match_threshold}\n"""

        r += f"""遮罩模式（X）: { mask_mode_dict[self.mask_mode] }\n"""

        if 'raw' not in self.mode:
            r += (f"""遮罩侵蚀度（W/S）: {self.erode_mask_modifier}\n"""
                  f"""遮罩边缘模糊度（E/D）: {self.blur_mask_modifier}\n"""
                  f"""运动模糊度（R/F）: {self.motion_blur_power}\n""")

        r += f"""输出面部大小（U/J）: {self.output_face_scale}\n"""

        if 'raw' not in self.mode:
            r += f"""色彩转换模式（C）: {ctm_dict[self.color_transfer_mode]}\n"""
            r += super().to_string(filename)

        r += f"""超分辨率强度（T/G）: {self.super_resolution_power}\n"""

        if 'raw' not in self.mode:
            r += (f"""降噪数值（I/K）: {self.image_denoise_power}\n"""
                  f"""插值数值（O/L）: {self.bicubic_degrade_power}\n"""
                  f"""降质数值（P/;）: {self.color_degrade_power}\n""")

        r += "================"

        return r


class MergerConfigFaceAvatar(MergerConfig):

    def __init__(self, temporal_face_count=0,
                       add_source_image=False):
        super().__init__(type=MergerConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count

        #changeable params
        self.add_source_image = add_source_image

    def copy(self):
        return copy.copy(self)

    #override
    def ask_settings(self):
        self.add_source_image = io.input_bool("添加源图像以进行比较？", False, help_message="添加源图像以进行比较.")
        super().ask_settings()

    def toggle_add_source_image(self):
        self.add_source_image = not self.add_source_image

    #override
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfigFaceAvatar):
            return super().__eq__(other) and \
                   self.add_source_image == other.add_source_image

        return False

    #override
    def to_string(self, filename):
        return (f"合成配置 {filename}:\n"
                f"添加源图像: {self.add_source_image}\n") + \
                super().to_string(filename) + "================"

