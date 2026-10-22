"""Reconstruction network with a cosine-sine phase output."""

import torch
import torch.nn as nn

from .representations import InputRepresentation


def conv_encoder(in_channels, widths):
    """Three stride-2 5 x 5 convolutions with ReLU (H x W -> H/8 x W/8)."""

    layers, channels = [], in_channels
    for width in widths:
        layers += [
            nn.Conv2d(channels, width, 5, stride=2, padding=2),
            nn.ReLU(),
        ]
        channels = width
    return nn.Sequential(*layers)


def matched_widths(nconv=32):
    """Widths (a, 2a) of one encoder matched to two standard encoders.

    One encoder (a, 2a, 4 nconv) plus a 128 -> 128 fusion has about as many
    parameters as two encoders (nconv, 2 nconv, 4 nconv) plus the 256 -> 128
    fusion, so single-view inputs keep the parameter count of the model.
    """

    def parameters(widths, in_channels=1):
        total = 0
        for width in widths:
            total += in_channels * width * 25 + width
            in_channels = width
        return total

    target = 2 * parameters((nconv, 2 * nconv, 4 * nconv))
    target += (8 * nconv) * (4 * nconv)
    width = min(
        range(nconv, 4 * nconv + 1),
        key=lambda a: abs(
            parameters((a, 2 * a, 4 * nconv)) + (4 * nconv) ** 2 - target
        ),
    )
    return width, 2 * width


class Decoder(nn.Module):
    """Three upsampling blocks; the skip feature joins after block 1."""

    def __init__(self, channels=128, skip_channels=64, out_channels=1):
        super().__init__()
        half = channels // 2

        def conv(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

        self.block1 = nn.Sequential(
            conv(channels, channels),
            nn.ReLU(),
            conv(channels, channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            conv(channels + skip_channels, half),
            nn.ReLU(),
            conv(half, half),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            conv(half, half), nn.ReLU(), conv(half, half), nn.ReLU()
        )
        self.out = conv(half, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, features, skip):
        x = self.upsample(self.block1(features))
        x = torch.cat([x, skip], dim=1)
        x = self.upsample(self.block2(x))
        x = self.upsample(self.block3(x))
        return self.out(x)


class ReconstructionNet(nn.Module):
    """Direct reconstruction of amplitude and phase from one pattern.

    The input representation produces one or more views. Each view has its
    own encoder (or, with stack_views=True, all views are channels of one
    widened encoder); the encoder outputs are fused by a bias-free 1 x 1
    convolution. A skip path reads the input pattern. Three decoders predict
    the amplitude (sigmoid) and the phase coordinates (c, s) without output
    activation, which are projected onto the unit circle.
    """

    def __init__(self, representation, stack_views=False, nconv=32):
        super().__init__()
        self.representation = representation
        num_views = representation.num_views
        self.separate_encoders = num_views > 1 and not stack_views
        if self.separate_encoders:
            self.encoders = nn.ModuleList(
                [
                    conv_encoder(1, (nconv, 2 * nconv, 4 * nconv))
                    for _ in range(num_views)
                ]
            )
        else:
            if representation.widened or num_views > 1:
                widths = matched_widths(nconv)
            else:
                widths = (nconv, 2 * nconv)
            self.encoders = nn.ModuleList(
                [conv_encoder(num_views, (*widths, 4 * nconv))]
            )
        self.fusion = nn.Conv2d(
            len(self.encoders) * 4 * nconv, 4 * nconv, 1, bias=False
        )
        self.amplitude_decoder = Decoder(4 * nconv, 2 * nconv)
        self.cos_decoder = Decoder(4 * nconv, 2 * nconv)
        self.sin_decoder = Decoder(4 * nconv, 2 * nconv)
        self.skip = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(nconv, 2 * nconv, 3, stride=2, padding=1),
        )

    def forward(self, intensity):
        views = self.representation(intensity)
        if self.separate_encoders:
            features = torch.cat(
                [encoder(view) for encoder, view in zip(self.encoders, views)],
                dim=1,
            )
        else:
            features = self.encoders[0](torch.cat(views, dim=1))
        features = self.fusion(features)
        skip = self.skip(intensity)

        amplitude = torch.sigmoid(self.amplitude_decoder(features, skip))
        cos_phase = self.cos_decoder(features, skip)
        sin_phase = self.sin_decoder(features, skip)
        magnitude = torch.sqrt(cos_phase**2 + sin_phase**2 + 1e-8)
        return amplitude, cos_phase / magnitude, sin_phase / magnitude


def build_model(representation, calibration, stack_views=False):
    """Create the network for a representation name and calibration."""

    return ReconstructionNet(
        InputRepresentation(representation, calibration),
        stack_views=stack_views,
    )
