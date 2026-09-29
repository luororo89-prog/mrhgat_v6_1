import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from D_mrhgat.tests_posthoc import standardized_coefficient

def test_standardized_partial_coefficient_has_known_sign_and_scale():
    X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]]*4,dtype=float)
    y=-2*X[:,0]+3*X[:,1]
    expected=-2/np.sqrt(13)
    assert abs(standardized_coefficient(y,X)-expected)<1e-12

def test_undefined_response_is_not_counted_as_support():
    X=np.arange(40).reshape(20,2).astype(float)
    assert np.isnan(standardized_coefficient(np.ones(20),X))
