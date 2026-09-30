"""Reading video specs and extracting frames with ffmpeg."""
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".mts", ".m2ts"}
HDR_TRANSFERS = {"smpte2084": "HDR10/PQ", "arib-std-b67": "HLG"}


@dataclass
class VideoSpecs:
    path: str
    width: int          # as displayed (after rotation)
    height: int
    fps: float
    duration_s: float
    codec: str
    profile: str
    pix_fmt: str
    bit_depth: int
    rotation: int
    orientation: str
    color_transfer: str
    color_primaries: str
    color_space: str
    hdr: bool
    hdr_format: str     # "HLG", "HDR10/PQ", "Dolby Vision (HLG base)", or ""
    bitrate_mbps: float
    device: str

    def summary(self) -> str:
        res = f"{self.width}x{self.height}"
        hdr = f"YES ({self.hdr_format})" if self.hdr else "no (SDR)"
        return (
            f"  File:        {Path(self.path).name}\n"
            f"  Resolution:  {res} ({self.orientation}{', rotated ' + str(self.rotation) + ' deg' if self.rotation else ''})\n"
            f"  FPS:         {self.fps:.2f}\n"
            f"  Duration:    {self.duration_s:.1f} s\n"
            f"  Codec:       {self.codec} {self.profile} ({self.pix_fmt}, {self.bit_depth}-bit)\n"
            f"  Bitrate:     {self.bitrate_mbps:.1f} Mbps\n"
            f"  Color:       transfer={self.color_transfer or '?'} primaries={self.color_primaries or '?'}\n"
            f"  HDR:         {hdr}\n"
            f"  Device:      {self.device or 'unknown'}"
        )

    def to_dict(self):
        return asdict(self)


def _rate(s: str) -> float:
    if not s or s == "0/0":
        return 0.0
    num, _, den = s.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path) -> VideoSpecs:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    fmt = info.get("format", {})
    tags = {k.lower(): val for k, val in {**fmt.get("tags", {}), **v.get("tags", {})}.items()}

    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if "rotate" in tags:
        rotation = int(tags["rotate"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    transfer = v.get("color_transfer", "")
    dovi = any("DOVI" in sd.get("side_data_type", "") or "Dolby Vision" in sd.get("side_data_type", "")
               for sd in v.get("side_data_list", []))
    hdr = transfer in HDR_TRANSFERS
    hdr_format = HDR_TRANSFERS.get(transfer, "")
    if dovi:
        hdr = True
        hdr_format = f"Dolby Vision ({hdr_format or 'unknown'} base)"

    pix_fmt = v.get("pix_fmt", "")
    bit_depth = int(v.get("bits_per_raw_sample") or (10 if "10" in pix_fmt else 12 if "12" in pix_fmt else 8))
    duration = float(v.get("duration") or fmt.get("duration") or 0)
    bitrate = float(v.get("bit_rate") or fmt.get("bit_rate") or 0) / 1e6
    device = " ".join(filter(None, [tags.get("com.apple.quicktime.make", ""),
                                    tags.get("com.apple.quicktime.model", "")])) or tags.get("encoder", "")

    return VideoSpecs(
        path=str(path), width=w, height=h, fps=_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        duration_s=duration, codec=v.get("codec_name", ""), profile=v.get("profile", ""),
        pix_fmt=pix_fmt, bit_depth=bit_depth, rotation=rotation,
        orientation="portrait" if h > w else "landscape" if w > h else "square",
        color_transfer=transfer, color_primaries=v.get("color_primaries", ""),
        color_space=v.get("color_space", ""), hdr=hdr, hdr_format=hdr_format,
        bitrate_mbps=bitrate, device=device,
    )


def tonemap_filter(specs: VideoSpecs, curve: str = "mobius", white_nits: int = 203) -> str:
    """HDR (PQ or HLG, BT.2020) -> SDR BT.709 using zscale + tonemap.

    white_nits: HDR level that becomes SDR white. 203 is the BT.2408 reference white;
    lower it (e.g. 150) for brighter frames, at the cost of blowing out highlights sooner.
    mobius keeps mid-tones (skin) nearly unchanged and only rolls off highlights. hable is
    the common default but darkens everything by about a third, which reads as "washed out".
    """
    tin = specs.color_transfer if specs.color_transfer in HDR_TRANSFERS else "arib-std-b67"
    pin = specs.color_primaries or "bt2020"
    min_ = specs.color_space or "bt2020nc"
    return (
        f"zscale=tin={tin}:pin={pin}:min={min_}:t=linear:npl={white_nits},"
        "format=gbrpf32le,zscale=p=bt709,"
        f"tonemap=tonemap={curve}:desat=0,"
        "zscale=t=bt709:m=bt709:r=pc,format=yuvj420p"
    )


def extract_frames(specs: VideoSpecs, out_dir: Path, fps: float = 8.0, jpeg_q: int = 2,
                   tonemap: str = "mobius", white_nits: int = 203) -> list[Path]:
    """Extract frames at `fps` at full resolution. Returns the list of JPEG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = [f"fps={fps}"]
    if specs.hdr:
        vf.append(tonemap_filter(specs, tonemap, white_nits))
    else:
        vf.append("scale=out_range=pc,format=yuvj420p")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", specs.path,
        "-map", "0:v:0", "-vf", ",".join(vf),
        "-q:v", str(jpeg_q), "-qmin", "1",
        str(out_dir / "f_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("f_*.jpg"))


def frame_time(path: Path, fps: float) -> float:
    """Timestamp (seconds) of an extracted frame, from its 1-based index."""
    idx = int(path.stem.split("_")[1])
    return (idx - 1) / fps


def list_videos(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
