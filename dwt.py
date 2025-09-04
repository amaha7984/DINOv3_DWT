import torch
import torch.nn as nn
import ptwt
import pywt
import torch.nn.functional as F

# ======================== DWT Module ========================
class HaarTransform(nn.Module):
    def __init__(self, level=1, mode="symmetric", with_grad=False):
        super().__init__()
        self.wavelet = pywt.Wavelet("haar")
        self.level = level
        self.mode = mode
        self.with_grad = with_grad

    def forward(self, x):
        with torch.set_grad_enabled(self.with_grad):
            Yl, *Yh = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=self.level, mode=self.mode)
            if len(Yh) < 1 or len(Yh[0]) != 3:
                raise ValueError("DWT failed: not enough subbands.")
            xH, xV, xD = Yh[0]
            # print(f"[DWT] Shapes: Yl={Yl.shape}, xH={xH.shape}, xV={xV.shape}, xD={xD.shape}")

            Yl = F.interpolate(Yl, size=(224, 224), mode='bilinear', align_corners=False)
            xH = F.interpolate(xH, size=(224, 224), mode='bilinear', align_corners=False)
            xV = F.interpolate(xV, size=(224, 224), mode='bilinear', align_corners=False)
            xD = F.interpolate(xD, size=(224, 224), mode='bilinear', align_corners=False)
            return Yl, xH, xV, xD

