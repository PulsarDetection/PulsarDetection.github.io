from django.urls import path
from . import views

urlpatterns = [
    path('cnn-predict/', views.cnn_predict, name='cnn_predict'),
    path('ann-predict/', views.ann_predict, name='ann_predict'),
    path('merged-predict/', views.merged_predict, name='merged_predict'),
    path('phcx-predict/', views.phcx_predict, name='phcx_predict'),
    # path('phcx-check/', views.phcx_check, name='phcx_check'),
    # path('cnn-predict-onnx/', views.cnn_predict_onnx, name='cnn_predict_onnx'),
    # path('ann-predict-onnx/', views.ann_predict_onnx, name='ann_predict_onnx'),
    # path('merged-predict-onnx/', views.merged_predict_onnx, name='merged_predict_onnx'),
]
