import rasterio
import numpy as np
import cv2

# %% Standard parameters for Web map loading features


WFS_DICT = dict(service='WFS', 
                version="2.0.0", 
                request='GetFeature',
                # typeName=layer,
                outputFormat='application/gml+xml; version=3.2',
                srsname='urn:ogc:def:crs:EPSG::28992', 
                # bbox=bbox_str,
                # start_index = 0,
                # count=1000,
                # sortby='bag:identificatie'
                )


WCS_DICT = dict(format='GEOTIFF_FLOAT32',
                crs='urn:ogc:def:crs:EPSG::28992',
                resx=0.5,
                resy=0.5)

# import line_profiler
# profile = line_profiler.LineProfiler()

# #%% Universal download functions 


# def _download_wms(url, bbox, layers, save_dir, save_name, size=(1000, 1000), format='image/png'):
        
#     wms = WebMapService(url, version='1.3.0')
    
#     layers_str = '&'.join(layers)
#     styles = [list(wms[layer].styles.keys())[0] for layer in layers]
    
#     save_path = f'{save_dir}/save_name_{layers_str}.tif'.lstrip('/')
    
#     response = wms.getmap(
#         layers=layers,
#         styles=styles,
#         srs='urn:ogc:def:crs:EPSG::28992',
#         bbox=bbox,
#         size=(1000,1000),
#         transparent=False
#        )
    
#     # Image(response.read())
    
#     img = response.read()
    
#     with open(save_path, 'wb') as file:
#         file.write(img)
        
#     return img
    

# def _download_wcs(url, identifier, save_dir, bbox, zoomout=False, 
#                   force_download=False, **kwargs):
#     """Load and save data with WebCoverageService."""
    
#     save_path = f'{save_dir}/{identifier}.tif'.lstrip('/')
    
#     if zoomout:
#         bbox = boundingbox.zoomout(bbox)
    
#     try:
#         if force_download:
#             # or not os.path.isfile(save_path)
#             raise Exception
            
#         dsr = rasterio.open(save_path, driver="GTiff")
#         if _bbox2area(bbox, res=WCS_DICT['resx']) != (dsr.height * dsr.width):
#             raise ValueError('Found file has wrong area.')
        
#         return dsr, save_path
    
#     except:
#         wcs = WebCoverageService(url, version='1.0.0')
        
#         kwargs.update({'bbox': tuple(bbox),
#                        'identifier': identifier})
        
#         params = WCS_DICT.copy()
#         params.update(kwargs)
                
#         url_disp = {'/'.join(url.split('/')[2:5])}
#         print(f"Retreiving data using WCS from {url_disp}")
        
#         # download data
#         response = wcs.getCoverage(**params)
        
#         # write data to file
#         with open(save_path, 'wb') as file:
#             file.write(response.read())
        
#         # open DataSet reader
#         try:
#             dsr = rasterio.open(save_path, driver="GTiff")
#         except rasterio.RasterioIOError:
#             raise ValueError(response.read())
                
#         return dsr, save_path

# # def _download_wms(bbox, url, layers, ):
# #     wms = WebMapService(url, version='1.3.0');\
# #     response = wms.getmap(
# #         layers=layers,
# #         styles=styles,
# #         srs='urn:ogc:def:crs:EPSG::28992',
# #         # bbox=tuple(bbox_zoomout),
# #         bbox=bbox_zoomout,
# #         # width=400,
# #         # height=500,
# #         size=(1000,1000),
# #         # width = 1000,
# #         # height=1000,
# #         # size=(1000,1000),
# #         # format='image/GeoTIFF',
# #         format='image/png',
# #         # format='image/jpeg',
# #        transparent=False
# #        )
    
    
    
# #     Image(response.read())
    
# #     with open(save_path, 'wb') as file:
# #         file.write(response.read())

# def _download_wfs(bbox, url,  typeName, count=1000, zoomout=False, index=None,
#                   drop_cols=None, **kwargs):
#     """Download GeoDataFrame from online database."""
    
#     gdf = gpd.GeoDataFrame() # start with empty gdf
    
#     if zoomout:
#         bbox = boundingbox.zoomout(bbox)

#     # transform bbox if another crs than the standard crs is used
#     if 'srsname' in kwargs:
#         srsname = kwargs['srsname']
#         if srsname != WFS_DICT['srsname']:
#             bbox = crs_transformation(
#                 bbox, crs_in=WFS_DICT['srsname'], crs_out=srsname)
        
#             bbox = reverse_bbox_order(bbox)
#             bbox = tuple(bbox)

#     bbox_str = ','.join(map(str, bbox))
    
#     kwargs.update({'bbox': bbox_str,
#                    'typeName': typeName})
    
#     params = WFS_DICT.copy()
#     params.update(kwargs)
    
#     url_disp = {'/'.join(url.split('/')[2:5])}
#     print(f"Retreiving data using WFS from {url_disp}")
    
#     startindex = 0
#     while True:
#         params.update({'startindex': startindex})
        
#         # Parse the URL with parameters
#         q = Request('GET', url, params=params).prepare().url
        
#         # Read data from URL
#         gdf_snippet = gpd.read_file(q)
#         gdf = gdf.append(gdf_snippet)
        
#         if len(gdf_snippet) < count - 1:
#             break
        
#         startindex += count
        
#     if index:
#         gdf = gdf.set_index(index)
#         # gdf.index = pd.to_numeric(gdf.index)

#     if drop_cols:
#         gdf = gdf.drop(columns=drop_cols)

#     # transform coordinate system
#     if gdf.crs != WFS_DICT['srsname']:
#         gdf = gdf.to_crs(WFS_DICT['srsname'])

#     return gdf


# %% Rasterio functions




# %% GeoDataFrame functions







# %% Geotiff functions
 




# %% Address functions

def location2address_str(location,
                          address_setup=['town', 'road', 'house_number'],
                          delimiter=' '):
    address_str = ''
    for item in address_setup:
        if item in location.raw['address']:
            if item != address_setup[0]:
                address_str += delimiter
            address_str += str(location.raw['address'][item])
            
            if item == 'town':
                address_str += ','

    return address_str




# %% Coordinate system functions









# def _arr2rgb(arr, cm=plt.cm.get_cmap('viridis')):
#     """Normalize numpy array and return rgb image."""
    
#     normed_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#     mapped_data = cm(normed_data, bytes=True)[..., :3]

#     return mapped_data

# %% To order


def get_dominant_angle(arr):
    """Find the dominant angle over which a street-map is rotated."""
    arr_rgb = _arr2rgb(arr)
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2GRAY)

    canimg = cv2.Canny(gray, 50, 200)
    rho_step = .1
    rho = rho_step
    lines = cv2.HoughLines(canimg, rho, (180 / rho_step), 100)
    while lines is None:
        rho += rho_step
        lines = cv2.HoughLines(canimg, rho, np.pi / (180 / rho_step), 100)

    lines_lst = list(lines[:, :, 1].flatten())

    # Sort lines on most-occuring rotation angle
    dominant_angle_rad = sorted(lines_lst,
                                key=lines_lst.count,
                                reverse=True)[0]
    dominant_angle = np.rad2deg(dominant_angle_rad) % 90

    if dominant_angle > 60:
        dominant_angle = dominant_angle - 90

    print('To align the map with horizontal and vertical axis, rotate it over '
          + str(round(dominant_angle, 3)) + ' degrees')
    return round(dominant_angle, 2)













