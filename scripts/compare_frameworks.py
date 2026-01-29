import torch
import torch.nn as nn
import time
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        # Stem: Conv 7x7 S=2 -> BN -> ReLU -> MaxPool S=2 (compounded to S=4 for simplicity to match Tracea test)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def benchmark_mps(iters=50, warmup=10):
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return None

    device = torch.device("mps")
    model = ResNet18().half().to(device)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224).half().to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            torch.mps.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(x)
            torch.mps.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = np.mean(latencies)
    print(f"PyTorch (MPS) ResNet-18 FP16 Results:")
    print(f"  Avg Latency: {avg_latency:.3f} ms")
    print(f"  Throughput: {1000.0/avg_latency:.1f} img/s")
    
    return avg_latency

if __name__ == "__main__":
    benchmark_mps()
