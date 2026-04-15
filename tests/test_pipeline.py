import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.model_training import build_pipeline

def test_build_pipeline_structure():

    clf = DecisionTreeClassifier()
    pipeline = build_pipeline(clf)
    
    assert isinstance(pipeline, Pipeline), "Function must return an instance of sklearn.pipeline.Pipeline"
    
    assert 'preprocessor' in pipeline.named_steps, "Pipeline is missing the 'preprocessor' step"
    
    assert isinstance(pipeline.named_steps['preprocessor'], ColumnTransformer), "Preprocessor must be an instance of ColumnTransformer"
    
    assert 'classifier' in pipeline.named_steps, "Pipeline is missing the 'classifier' step"
