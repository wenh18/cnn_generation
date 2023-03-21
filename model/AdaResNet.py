from functools import partial
from re import L
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
import math
import torch.nn.functional as F
from .seq2seq import get_seq2seq_model
# from encoder import Seq2seqEncoder
# from decoder import Seq2seqDecoder
# import seq2seq
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model, model_entrypoint
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, \
    get_act_layer, get_norm_layer, create_classifier
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'Ada_ResNet50': _cfg(),
}

class FilterSelectModule(nn.Module):
    def __init__(self,selected_length,kernel_number):
        super().__init__()
        self.selected_length=selected_length
        self.kernel_number=kernel_number
        #self.tokenizer=BertTokenizer.from_pretrained('./bert-large-cased-whole-word-masking')
        #self.bert=BertModel.from_pretrained('./bert-large-cased-whole-word-masking').cuda()
        #self.bert=self.bert.to("cuda")

        #for param in self.bert.parameters():
        #    param.requires_grad=False

        self.lstm = nn.LSTM(input_size=100, hidden_size=kernel_number, num_layers=2).cuda()

    def forward(self,prompt):
        prompt=prompt.float()
        #prompt ：tuple
        #encoded_input=self.tokenizer(prompt,return_tensors='pt',padding=True)
        #encoded_input=encoded_input.to("cuda")
        #output=self.bert(**encoded_input)
        #output[0] : [batch_size,length,1024]
        #cls=output[0][:,0,:]
        #cls: [length,batch_size,1024]
        prompt=prompt.unsqueeze(1)
        prompt=prompt.repeat(1,self.selected_length,1)
        #cls=cls+get_sinusoid_encoding(self.selected_length,1024).to("cuda")
        prompt=prompt.permute(1,0,2)
        output,(h,c)=self.lstm(prompt)
        output=output.permute(1,0,2)
        output=Improved_SemHash(output)

        return output
def Improved_SemHash(select_weight):
    # select_weight:[batch_size,length,kernel_number]
    binary_selection = torch.lt(torch.zeros_like(select_weight), select_weight).float()
    gradient_selection = torch.max(torch.zeros_like(select_weight), torch.min(torch.ones_like(select_weight), (
                1.2 * torch.sigmoid(select_weight) - 0.1)))
    d = binary_selection + gradient_selection - gradient_selection.detach()
    return d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path=None,
    ):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path=None,
    ):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x, kernel_selection=None):
        shortcut = x
        if kernel_selection == None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.drop_block(x)
            x = self.act2(x)

            x = self.conv3(x)
            x = self.bn3(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            if self.downsample is not None:
                shortcut = self.downsample(shortcut)
            x += shortcut
            x = self.act3(x)
        else:
            x = self.conv1(x)
            x = x * kernel_selection[:, 0, :][:, :self.conv1.out_channels, None, None]
            x = self.bn1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = x * kernel_selection[:, 1, :][:, :self.conv2.out_channels, None, None]
            x = self.bn2(x)
            x = self.drop_block(x)
            x = self.act2(x)

            x = self.conv3(x)
            x = x * kernel_selection[:, 2, ][:, :self.conv3.out_channels, None, None]
            x = self.bn3(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            if self.downsample is not None:
                shortcut = self.downsample(shortcut)
            x += shortcut*kernel_selection[:, 2, ][:, :self.conv3.out_channels, None, None]
            x = self.act3(x)

        return x


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def downsample_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn,
        channels,
        block_repeats,
        inplanes,
        reduce_first=1,
        output_stride=32,
        down_kernel_size=1,
        avg_down=False,
        drop_block_rate=0.,
        drop_path_rate=0.,
        **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.ModuleList(blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class Ada_ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering
    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.
    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample
    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled
    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled
    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block,
            layers,
            num_classes=100,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            ada_kernel=False
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(Ada_ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        # self.grad_checkpointing = False
        self.ada_kernel = ada_kernel

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # 创建model时要让ada_kernel为true
        if self.ada_kernel:
            '''
            encoder = Seq2seqEncoder(hidden_dim=1024,
                                     bidirectional=True, )
            # 为了方便产生的output不需要embedding就可以循环使用，hidden_dim和Num_classes都是2048，也可以把seq2seq里的project层去掉
            decoder = Seq2seqDecoder(hidden_dim=512 * block.expansion, num_classes=512 * block.expansion,
                                     max_decoding_step=sum(layers)* 3)
            self.seq2seq = Seq2seq(encoder, decoder)
            '''
            # self.lstm=FilterSelectModule(selected_length=sum(layers)* 3,kernel_number=512 * block.expansion)
            self.lstm=get_seq2seq_model()
        inplanes = 64

        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.embedding=nn.Embedding(100,100)
        infeatures=self.fc.in_features+10*100
        outfeatures=self.fc.out_features
        self.fc2=nn.Linear(in_features=infeatures,out_features=outfeatures,bias=True)


        self.init_weights(zero_init_last=zero_init_last)

    @staticmethod
    def from_pretrained(model_name: str, load_weights=True, **kwargs) -> 'Ada_ResNet':
        entry_fn = model_entrypoint(model_name, 'Ada_Resnet')
        return entry_fn(pretrained=not load_weights, **kwargs)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x, decision=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.maxpool(x)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        # x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)

        if decision == None:
            for layer in self.layer1:
                x = layer(x)
            for layer in self.layer2:
                x = layer(x)
            for layer in self.layer3:
                x = layer(x)
            for layer in self.layer4:
                x = layer(x)
        else:
            gate_num = 0
            for layer in self.layer1:
                x = layer(x, decision[:, gate_num:gate_num + 3, :])
                gate_num += 3

            for layer in self.layer2:
                x = layer(x, decision[:, gate_num:gate_num + 3, :])
                gate_num += 3

            for layer in self.layer3:
                x = layer(x, decision[:, gate_num:gate_num + 3, :])
                gate_num += 3

            for layer in self.layer4:
                x = layer(x, decision[:, gate_num:gate_num + 3, :])
                gate_num += 3

        return x

    def forward_head(self, x,embed, pre_logits: bool = False):
        #embed=embed.type(torch.LongTensor)
        embedding=self.embedding(embed)
        embedding=embedding.view(embedding.size(0),-1)
        x = self.global_pool(x)
        x=torch.cat((x,embedding),dim=1)
       # if self.drop_rate:
            #x = F.dropout(x, p=float(self.drop_rate), training=self.training)

        #x=x.unsqueeze(-1)

        #out=torch.bmm(embedding,x.view(x.size(0),x.size(1),1)).squeeze()
        # x if pre_logits else self.fc2(x)
        return self.fc2(x)

    def forward(self, x, prompt,embed):
        # prompt: batch_size * number_of_classes, eg: 64*100
        # embed: batch_size * number_of_classes_in_one_task eg: 64*10
        # decision: batch_size * number_of_kernels * maximum_number_of_channel_in_one_kernel eg: 64*48*2048
        # import pdb;pdb.set_trace()
        task = F.one_hot(embed, num_classes=100).to(torch.float32)
        decision = None
        if self.ada_kernel:
            # input应为【batchsize，length,1],定长
            #decision = self.seq2seq(inputs=prompt)
            decision=self.lstm(task)
            #decision = torch.stack(decision[0]).permute(1, 0, 2)
            #decision = Improved_SemHash(decision)
            # decision: 长度为length的列表，每一个元素都是tensor(batch_size,2048)
            # 后转为【batch，length，2048】
        x = self.forward_features(x, decision)
        x = self.forward_head(x,embed)
        return x, decision


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(Ada_ResNet, variant, pretrained, **kwargs)



def ada_ResNet50(pretrained=False, ada_kernel=True, **kwargs):
    """Constructs a Ada_ResNet50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], ada_kernel=ada_kernel,num_classes=10, **kwargs)
    return _create_resnet('Ada_ResNet50', pretrained, **dict(model_args, **kwargs))
