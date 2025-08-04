"""Interface for TESS-EBs catalog."""

import os
import logging
import numpy
import pandas
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent / "tess_ebs_catalog.csv"

logger = logging.getLogger(__name__)


class TESSEBsCatalog:
    
    def __init__(self, catalog_path=None):
        if catalog_path is None:
            catalog_path = CATALOG_PATH
            
        self.catalog_path = catalog_path
        self.catalog_data = None
        self.load_catalog()
    
    def load_catalog(self):
        if not os.path.exists(self.catalog_path):
            logger.warning(f"Catalog file not found at {self.catalog_path}")
            self.catalog_data = None
            return
            
        try:
            self.catalog_data = pandas.read_csv(self.catalog_path)
            logger.info(f"Loaded TESS-EBs catalog with {len(self.catalog_data)} entries")
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            self.catalog_data = None
    
    def is_available(self):
        return self.catalog_data is not None
    
    def get_eb_info(self, tic_id):
        if not self.is_available():
            return None
            
        matches = self.catalog_data[self.catalog_data['tess_id'] == tic_id]
        if len(matches) == 0:
            return None
            
        eb_data = matches.iloc[0]
        
        result = {
            'tic_id': int(eb_data['tess_id']),
            'period': float(eb_data['period']),
            'period_uncert': float(eb_data['period_uncert']) if pandas.notna(eb_data['period_uncert']) else None,
            'bjd0': float(eb_data['bjd0']),
            'bjd0_uncert': float(eb_data['bjd0_uncert']) if pandas.notna(eb_data['bjd0_uncert']) else None,
            'morph_coeff': float(eb_data['morph_coeff']) if pandas.notna(eb_data['morph_coeff']) else None,
            'ra': float(eb_data['ra']),
            'dec': float(eb_data['dec']),
            'tmag': float(eb_data['Tmag']) if pandas.notna(eb_data['Tmag']) else None,
        }
        
        if pandas.notna(eb_data['prim_depth_pf']):
            result['prim_depth_pf'] = float(eb_data['prim_depth_pf'])
        if pandas.notna(eb_data['prim_width_pf']):
            result['prim_width_pf'] = float(eb_data['prim_width_pf'])
        if pandas.notna(eb_data['prim_pos_pf']):
            result['prim_pos_pf'] = float(eb_data['prim_pos_pf'])
            
        if pandas.notna(eb_data['sec_depth_pf']):
            result['sec_depth_pf'] = float(eb_data['sec_depth_pf'])
        if pandas.notna(eb_data['sec_width_pf']):
            result['sec_width_pf'] = float(eb_data['sec_width_pf'])
        if pandas.notna(eb_data['sec_pos_pf']):
            result['sec_pos_pf'] = float(eb_data['sec_pos_pf'])
            
        for param in ['prim_depth_2g', 'prim_width_2g', 'prim_pos_2g', 
                      'sec_depth_2g', 'sec_width_2g', 'sec_pos_2g']:
            if pandas.notna(eb_data[param]):
                result[param] = float(eb_data[param])
                
        if pandas.notna(eb_data['sectors']):
            result['sectors'] = eb_data['sectors']
            
        return result
    
    def get_period_range(self, tic_id, factor=0.1):
        eb_info = self.get_eb_info(tic_id)
        if eb_info is None:
            return None
            
        period = eb_info['period']
        period_uncert = eb_info.get('period_uncert', None)
        
        if period_uncert is not None and period_uncert > 0:
            range_size = max(3 * period_uncert, factor * period)
        else:
            range_size = factor * period
            
        min_p = max(0.1, period - range_size)
        max_p = min(100.0, period + range_size)
        
        return (min_p, max_p)
    
    def get_all_tic_ids(self):
        if not self.is_available():
            return []
        return self.catalog_data['tess_id'].tolist()
    
    def search_by_period(self, min_period, max_period):
        if not self.is_available():
            return []
            
        mask = (self.catalog_data['period'] >= min_period) & (self.catalog_data['period'] <= max_period)
        return self.catalog_data[mask]['tess_id'].tolist()


catalog_instance = None

def get_catalog():
    global catalog_instance
    if catalog_instance is None:
        catalog_instance = TESSEBsCatalog()
    return catalog_instance


def get_eb_info(tic_id):
    return get_catalog().get_eb_info(tic_id)


def get_catalog_period_range(tic_id, factor=0.1):
    return get_catalog().get_period_range(tic_id, factor)


def is_in_catalog(tic_id):
    return get_eb_info(tic_id) is not None 