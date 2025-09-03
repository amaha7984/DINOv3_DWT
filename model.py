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

class DINOv3WithDWT(nn.Module):
    def __init__(self, num_classes=2, use_all_bands=True,
                 repo_dir=".../dinov3", #clone official dinov3
                 weights=".../weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", #download weight with process mentioned in DinoV3 official repository
                 stats="SAT", freeze_backbone=True):
        super().__init__()
        self.dwt = HaarTransform()
        self.use_all_bands = use_all_bands

        
        self.backbone = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=weights)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
            self.backbone.eval()


        self.feat_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)
        assert self.feat_dim is not None, "Cannot infer feature dim; ensure DINOv3 Hub model exposes embed_dim."

        self.classifier = nn.Linear(self.feat_dim * (4 if use_all_bands else 1), num_classes)

    def _norm(self, x):
        mean = x.new_tensor(self.mean)[None,:,None,None]
        std  = x.new_tensor(self.std )[None,:,None,None]
        return (x - mean) / std

    def _embed(self, x):
        # Prefer forward_features → class token
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]     # [B, feat_dim]
    
        y = self.backbone(x)
        if torch.is_tensor(y) and y.dim() == 2:
            return y
        raise RuntimeError("Unexpected backbone output; forward_features lacks 'x_norm_clstoken' and forward() didn't return [B, D].")

    def forward(self, x):
        Yl, xH, xV, xD = self.dwt(x)
        Yl = self._norm(Yl)
        feat_Yl = self._embed(Yl)
        if self.use_all_bands:
            feats = torch.cat([feat_Yl,
                               self._embed(self._norm(xH)),
                               self._embed(self._norm(xV)),
                               self._embed(self._norm(xD))], dim=1)
        else:
            feats = feat_Yl
        return self.classifier(feats)

