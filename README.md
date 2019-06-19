# caffe_pytorch_m2det

运行python3 run.py


大多借鉴于：https://github.com/starimeL/PytorchConverter

但由于原作者不是基于ssd caffe，有少许需要改动，主要是增加permute层

使用作者这个源码进行转换，若caffe源环境不同，需要：

1\编译好自己使用的caffe，如ssd caffe，然后将caffe_pb2.py对源码进行替换。

2\参考ConvertLayer_caffe.py中别的层实现方式，实现permute层：

def Permute(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = 'Permute'
    assert len(pytorch_layer.rev_dim_indices) == 4, len(pytorch_layer.rev_dim_indices)
    assert pytorch_layer.rev_dim_indices[0] == 0, pytorch_layer.rev_dim_indices[0]
    layer.permute_param.order.extend([0,2,3,1])
    return layer
并在def build_converter(opts):中增加

3\更改别的numpy()为.cpu().numpy()

ps:必须用python3,pytorch0.2.0,caffe需要支持使用层的源
