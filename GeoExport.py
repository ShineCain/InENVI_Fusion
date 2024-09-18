from osgeo import gdal
import numpy as np

def Write_Envi(data,geotrans,proj,path):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'float16' in data.dtype.name:
        datatype = gdal.GDT_Float32#GDT_Ufloat16
    elif 'complex64' in data.dtype.name:
        datatype = gdal.GDT_CFloat32
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        bands, height, width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        bands, height, width = data.shape
    #创建文件
    driver = gdal.GetDriverByName("ENVI")
    dataset = driver.Create(path, int(width), int(height), int(bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(geotrans) #写入仿射变换参数
        dataset.SetProjection(proj) #写入投影
    for i in range(bands):
        dataset.GetRasterBand(i+1).WriteArray(data[i])
    del dataset


def Write_GTIF(data,geotrans,proj,path):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'float16' in data.dtype.name:
        datatype = gdal.GDT_Float32#GDT_Ufloat16
    elif 'complex64' in data.dtype.name:
        datatype = gdal.GDT_CFloat32
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        bands, height, width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        bands, height, width = data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(width), int(height), int(bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(geotrans) #写入仿射变换参数
        dataset.SetProjection(proj) #写入投影
    for i in range(bands):
        dataset.GetRasterBand(i+1).WriteArray(data[i])
    del dataset

