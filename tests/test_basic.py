#!/usr/bin/env python3
"""
Test suite for Premier League Transfer Market Value Predictor.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

class TestTransferValuePredictor(unittest.TestCase):
    
    def test_file_creation(self):
        """Test that required files and directories are present."""
        # Test main files exist
        self.assertTrue(os.path.exists('collect_data.py'))
        self.assertTrue(os.path.exists('train_model.py'))
        self.assertTrue(os.path.exists('main.py'))
        self.assertTrue(os.path.exists('requirements.txt'))
        
        # Test directories exist
        self.assertTrue(os.path.exists('data/raw'))
        self.assertTrue(os.path.exists('data/processed'))
        self.assertTrue(os.path.exists('models'))
        self.assertTrue(os.path.exists('src/data_collection'))
        self.assertTrue(os.path.exists('src/preprocessing'))
        self.assertTrue(os.path.exists('src/features'))
        self.assertTrue(os.path.exists('src/modeling'))
        self.assertTrue(os.path.exists('src/prediction'))
        self.assertTrue(os.path.exists('src/visualization'))
        self.assertTrue(os.path.exists('src/utils'))
        
    def test_requirements_content(self):
        """Test that requirements file contains necessary packages."""
        with open('requirements.txt', 'r') as f:
            content = f.read()
            
        required_packages = ['requests', 'pandas', 'scikit-learn', 'matplotlib', 'numpy']
        for package in required_packages:
            self.assertIn(package, content)
            
    def test_structure(self):
        """Test that the basic project structure is correct."""
        # Test key directory structure
        expected_dirs = [
            'data/raw',
            'data/processed', 
            'models',
            'src/data_collection',
            'src/preprocessing',
            'src/features',
            'src/modeling',
            'src/prediction',
            'src/visualization',
            'src/utils'
        ]
        
        for directory in expected_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} does not exist")

if __name__ == '__main__':
    unittest.main()