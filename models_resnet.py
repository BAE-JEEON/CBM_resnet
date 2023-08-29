
from CUB.template_model_resnet import MLP, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, End2EndModel


# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, arch, num_classes, n_attributes, expand_dim, three_class):
    arch = globals().get(arch)
    return arch(pretrained=pretrained, freeze=freeze, num_classes=num_classes, 
                        n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                        three_class=three_class)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, arch, num_classes, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    arch = globals().get(arch)
    model1 = arch(pretrained=pretrained, freeze=freeze, num_classes=num_classes, 
                          n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3))
    if n_class_attr == 3:
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

# Standard Model
def ModelXtoY(pretrained, freeze, arch, num_classes):
    arch = globals().get(arch)
    return arch(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux)

# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, n_attributes, three_class, connect_CY):
    return arch(pretrained=pretrained, freeze=freeze, num_classes=num_classes, 
                        n_attributes=n_attributes, bottleneck=False, three_class=three_class,
                        connect_CY=connect_CY)
