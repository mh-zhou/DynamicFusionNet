import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import collections
import threading
import queue
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from thop import profile, clever_format
from torchsummary import summary


class FutureResult:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result hasn't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res

_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])

class SlavePipe(_SlavePipeBase):
    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret

class SyncMaster:
    def __init__(self, master_callback):
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        if self._activated:
            assert self._queue.empty(), "Queue not clean before next init."
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = future
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        self._activated = True
        intermediates = [(0, master_msg)]
        for _ in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, "First result must be master."
        for i, msg in results:
            if i != 0:
                self._registry[i].put(msg)
        for _ in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def _sum_ft(x):
    return x.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(x):
    return x.unsqueeze(0).unsqueeze(-1)

_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias,
                self.training, self.momentum, self.eps
            )
        shape = input.size()
        x = input.view(shape[0], self.num_features, -1)
        size = x.size(0) * x.size(2)
        s = _sum_ft(x)
        ss = _sum_ft(x ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(s, ss, size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(s, ss, size))
        if self.affine:
            out = (x - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            out = (x - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return out.view(shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if copy_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [item for sub in intermediates for item in sub[1][:2]]
        gpus = [i[1].sum.get_device() for i in intermediates]
        size = sum(i[1].sum_size for i in intermediates)
        s, ss = ReduceAddCoalesced.apply(gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(s, ss, size)
        bcast = Broadcast.apply(gpus, mean, inv_std)
        out = []
        for idx, rec in enumerate(intermediates):
            out.append((rec[0], _MasterMessage(*bcast[idx*2:idx*2+2])))
        return out

    def _compute_mean_std(self, s, ss, size):
        assert size > 1, "BatchNorm requires size>1."
        mean = s / size
        var = (ss - s * mean) / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        return mean, var.clamp(self.eps) ** -0.5

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (got {input.dim()}D)")
        super()._check_input_dim(input)


BatchNorm2d = SynchronizedBatchNorm2d


class AtrousFeatureExtractor(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size,
                              stride=1, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GlobalLocalFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3, padding=1,
                                    groups=in_channels, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.global_pool(x).expand_as(x)
        l = self.local_conv(x)
        att = self.fuse(torch.cat([g, l], dim=1))
        return x * att


class ContextualFeatureRefinement(nn.Module):
    def __init__(self, backbone='256', output_stride=8, BatchNorm=BatchNorm2d):
        super().__init__()
        inplanes = 256 if backbone=='256' else 512
        dilations = [1,2,3,4] if output_stride==8 else [1,6,12,18]

        self.branches = nn.ModuleList([
            AtrousFeatureExtractor(inplanes, 512, 1,   0,           dilations[0], BatchNorm),
            AtrousFeatureExtractor(inplanes, 512, 3,   dilations[1], dilations[1], BatchNorm),
            AtrousFeatureExtractor(inplanes, 512, 3,   dilations[2], dilations[2], BatchNorm),
            AtrousFeatureExtractor(inplanes, 512, 3,   dilations[3], dilations[3], BatchNorm),
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(inplanes,512,1,bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(512*5,512,1,bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            GlobalLocalFusion(512),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=res[0].shape[2:], mode='bilinear', align_corners=True)
        res.append(gp)
        return self.project(torch.cat(res, dim=1))


class ResidualConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class ScaleUpProjector(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DynamicDenseFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = Conv_Block(in_ch,      out_ch)
        self.c2 = Conv_Block(out_ch+in_ch, out_ch)
        self.c3 = Conv_Block(out_ch*2+in_ch, out_ch)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(torch.cat([x1, x], dim=1))
        x3 = self.c3(torch.cat([x1, x, x2], dim=1))
        return x3


class CrossScaleAttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class PredictionHead(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 7, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DynamicFusionNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.FeatureStem = nn.Conv2d(img_ch, 32, 7, padding=3, bias=True)
        self.down1 = DynamicDenseFusion(32,  64)
        self.down2 = DynamicDenseFusion(64,  64)
        self.down3 = DynamicDenseFusion(64, 128)
        self.down4 = ResidualConvUnit(128,256)

        self.ContextualFeatureRefinement = ContextualFeatureRefinement('256', 8, BatchNorm2d)
        self.drop5 = nn.Dropout(0.2)

        self.up6   = ScaleUpProjector(512, 512)
        self.att4  = CrossScaleAttentionGate(512, 256, 256)
        self.dec6  = DynamicDenseFusion(512+256, 256)

        self.up7   = ScaleUpProjector(256, 256)
        self.att3  = CrossScaleAttentionGate(256, 128, 128)
        self.dec7  = DynamicDenseFusion(256+128, 128)

        self.up8   = ScaleUpProjector(128, 128)
        self.att2  = CrossScaleAttentionGate(128,  64,  64)
        self.dec8  = DynamicDenseFusion(128+64,   64)

        self.up9   = ScaleUpProjector(64, 64)
        self.att1  = CrossScaleAttentionGate(64,   64,  32)
        self.dec9  = DynamicDenseFusion(64+64,    64)

        self.PredictionHead = PredictionHead(64, 32, output_ch)

    def forward(self, x):
        x0 = self.FeatureStem(x)
        d1 = self.down1(x0); p1 = self.pool(d1)
        d2 = self.down2(p1); p2 = self.pool(d2)
        d3 = self.down3(p2); p3 = self.pool(d3)
        d4 = self.down4(p3); p4 = self.pool(d4)

        b  = self.drop5(self.ContextualFeatureRefinement(p4))

        u6 = self.up6(b)
        x6 = self.att4(u6, d4)
        u6 = self.dec6(torch.cat([x6, u6], dim=1))

        u7 = self.up7(u6)
        x7 = self.att3(u7, d3)
        u7 = self.dec7(torch.cat([x7, u7], dim=1))

        u8 = self.up8(u7)
        x8 = self.att2(u8, d2)
        u8 = self.dec8(torch.cat([x8, u8], dim=1))

        u9 = self.up9(u8)
        x9 = self.att1(u9, d1)
        u9 = self.dec9(torch.cat([x9, u9], dim=1))

        return self.PredictionHead(u9)

if __name__ == "__main__":
    num_classes = 2
    input_shape = (1, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynamicFusionNet(img_ch=input_shape[0], output_ch=num_classes).to(device)
    summary(model, input_shape)

    dummy = torch.randn(1, *input_shape).to(device)
    flops, params = profile(model, (dummy,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Total GFLOPS: {flops}")
    print(f"Total parameters: {params}")
