import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

# Path to the temporary images folder
TEMP_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'temporary')
# TEMP_IMAGE_PATH = os.path.join(TEMP_IMAGES_DIR, 'temp_image.png')

def resize_and_combine_to_3channel(pdm_plane, subbands, subints):
    pdm_plane_resized = cv2.resize(pdm_plane, (32, 32))
    subbands_resized = cv2.resize(subbands, (32, 32))
    subints_resized = cv2.resize(subints, (32, 32))
    
    pdm_plane_normalized = cv2.normalize(pdm_plane_resized, None, 0, 255, cv2.NORM_MINMAX)
    subbands_normalized = cv2.normalize(subbands_resized, None, 0, 255, cv2.NORM_MINMAX)
    subints_normalized = cv2.normalize(subints_resized, None, 0, 255, cv2.NORM_MINMAX)

    combined_image = np.stack((pdm_plane_normalized, subbands_normalized, subints_normalized), axis=-1)
    
    return combined_image

def readDataBlock(xmlnode):
    """ Turn any 'DataBlock' XML node into a numpy array of floats """
    vmin = float(xmlnode.get('min'))
    vmax = float(xmlnode.get('max'))
    string = xmlnode.text
    string = re.sub(r"[\t\s\n]", "", string)
    data = np.asarray(bytearray.fromhex(string), dtype=float)
    return data * (vmax - vmin) / 255. + vmin

class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path. """
        xmlroot = ET.parse(fname).getroot()
        
        opt_section = next(section for section in xmlroot.findall('Section') if 'pdmp' in section.get('name').lower())
        
        pdmNode = opt_section.find('SnrBlock')
        dm_index = np.asarray(list(map(float, pdmNode.find('DmIndex').text.split())))
        period_index = np.asarray(list(map(float, pdmNode.find('PeriodIndex').text.split())))
        period_index /= 1.0e12  # Picoseconds to seconds

        pdm_plane = readDataBlock(pdmNode.find('DataBlock')).reshape(dm_index.size, period_index.size)
        
        subintsNode = opt_section.find('SubIntegrations')
        self.subints = readDataBlock(subintsNode).reshape(int(subintsNode.get('nSub')), int(subintsNode.get('nBins')))
        
        subbandsNode = opt_section.find('SubBands')
        self.subbands = readDataBlock(subbandsNode).reshape(int(subbandsNode.get('nSub')), int(subbandsNode.get('nBins')))
        
        self.combined_image = resize_and_combine_to_3channel(pdm_plane, self.subbands, self.subints)

def process_file(fname):
    cand = Candidate(fname)
    combined_image = cand.combined_image

    fig, ax = plt.subplots()
    ax.imshow(combined_image.astype(np.uint8))
    ax.axis('off')

    # Save the image to the temporary directory
    # plt.savefig(TEMP_IMAGES_DIR, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig('temporary/temp_image.png', format='png', bbox_inches='tight', pad_inches=0)
    # print('data',image)
    plt.close(fig)
    return 'temporary/temp_image.png'

# process_file('check.phcx')