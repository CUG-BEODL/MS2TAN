import os

import numpy as np
from osgeo import gdal
from PIL import Image, ImageEnhance, ImageOps


# 读取tiff文件
def readGeoTIFF(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据

    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    return im_data, im_geotrans, im_proj


# 写入tiff文件
def CreateGeoTiff(outRaster, image, geo_transform, projection):
    no_bands = 0
    rows = 0
    cols = 0

    driver = gdal.GetDriverByName("GTiff")
    if len(image.shape) == 2:
        no_bands = 1
        rows, cols = image.shape
    elif len(image.shape) == 3:
        no_bands, rows, cols = image.shape

    # DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Byte)
    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Float32)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    if no_bands == 1:
        DataSet.GetRasterBand(1).WriteArray(image)  # 写入数组数据
    else:
        for i in range(no_bands):
            DataSet.GetRasterBand(i + 1).WriteArray(image[i])
    del DataSet


def compress(path, method="LZW"):
    """使用gdal进行文件压缩，
    LZW方法属于无损压缩"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName("GTiff")
    target_path = path.replace(".tif", "_temp.tif")
    driver.CreateCopy(
        target_path,
        dataset,
        strict=1,
        options=["TILED=YES", "COMPRESS={0}".format(method)],
    )
    os.remove(path)
    os.rename(target_path, path)
    del dataset


"""将影像合成真彩影像"""


def linear(arr):
    arr_min, arr_max = arr.min(), arr.max()
    arr = (arr - arr_min) / (arr_max - arr_min) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


def percent_linear(arr, percent=2):
    arr_min, arr_max = np.percentile(arr, (percent, 100 - percent))
    arr = (arr - arr_min) / (arr_max - arr_min) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


def optimized_linear(arr):
    a, b = np.percentile(arr, (2.5, 99))
    c = a - 0.1 * (b - a)
    d = b + 0.5 * (b - a)
    arr = (arr - c) / (d - c) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


def rgbimg(input_path):
    data, geo, pro = readGeoTIFF(input_path)
    rgb = data[:3, :, :][::-1]  # 取前三个波段为RGB
    rgb = optimized_linear(rgb)
    imageOriginal = Image.fromarray(np.uint8(rgb.transpose(1, 2, 0)))
    # 直方图均衡
    imageOriginal = ImageOps.equalize(imageOriginal)
    # 亮度调整
    imageOriginal = ImageEnhance.Brightness(imageOriginal).enhance(0.9)
    # 对比度调整
    imageOriginal = ImageEnhance.Contrast(imageOriginal).enhance(0.9)
    # 饱和度调整
    imageOriginal = ImageEnhance.Color(imageOriginal).enhance(1.1)
    # 清晰度调整
    imageOriginal = ImageEnhance.Sharpness(imageOriginal).enhance(1.5)

    imageOriginal.show()
