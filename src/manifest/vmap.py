"""VMAP 1.0 generation with inline VAST creatives."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


VMAP_NS = "http://www.iab.net/videosuite/vmap"

ET.register_namespace("vmap", VMAP_NS)


def qname(name: str) -> str:
    return f"{{{VMAP_NS}}}{name}"


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    if seconds < 0:
        raise ValueError("Time cannot be negative")

    milliseconds = int(round(seconds * 1000))

    hours, remainder = divmod(
        milliseconds,
        3_600_000,
    )

    minutes, remainder = divmod(
        remainder,
        60_000,
    )

    secs, millis = divmod(
        remainder,
        1000,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{millis:03d}"
    )


def build_inline_vast(
    *,
    break_id: str,
    brand_id: str,
    brand_name: str,
    creative_uri: str,
    ad_duration_seconds: float,
) -> ET.Element:
    """
    Build a minimal inline VAST 3.0 creative.

    The hackathon demo uses synthetic creative URIs.
    """
    vast = ET.Element(
        "VAST",
        {
            "version": "3.0",
        },
    )

    ad = ET.SubElement(
        vast,
        "Ad",
        {
            "id": (
                f"{brand_id}-{break_id}"
            ),
        },
    )

    inline = ET.SubElement(
        ad,
        "InLine",
    )

    ad_system = ET.SubElement(
        inline,
        "AdSystem",
        {
            "version": "1.0",
        },
    )

    ad_system.text = (
        "Hoichoi Ad Context Demo"
    )

    title = ET.SubElement(
        inline,
        "AdTitle",
    )

    title.text = brand_name

    impression = ET.SubElement(
        inline,
        "Impression",
        {
            "id": (
                f"imp-{break_id}"
            ),
        },
    )

    impression.text = (
        "urn:hoichoi-ad-context:"
        f"impression:{break_id}"
    )

    creatives = ET.SubElement(
        inline,
        "Creatives",
    )

    creative = ET.SubElement(
        creatives,
        "Creative",
    )

    linear = ET.SubElement(
        creative,
        "Linear",
    )

    duration = ET.SubElement(
        linear,
        "Duration",
    )

    duration.text = format_time(
        ad_duration_seconds
    )

    media_files = ET.SubElement(
        linear,
        "MediaFiles",
    )

    media_file = ET.SubElement(
        media_files,
        "MediaFile",
        {
            "delivery": "progressive",
            "type": "video/mp4",
            "width": "1280",
            "height": "720",
        },
    )

    media_file.text = creative_uri

    return vast


def build_vmap(
    *,
    video_name: str,
    breaks: list[dict],
    ad_duration_seconds: float,
) -> ET.ElementTree:
    """
    Build one VMAP document for one content video.
    """
    root = ET.Element(
        qname("VMAP"),
        {
            "version": "1.0",
        },
    )

    ordered = sorted(
        breaks,
        key=lambda item: (
            float(
                item[
                    "timestamp_seconds"
                ]
            )
        ),
    )

    for item in ordered:
        selected_brand = (
            item.get(
                "selected_brand"
            )
        )

        if selected_brand is None:
            raise ValueError(
                f"{video_name}: "
                "cannot create VMAP "
                "without selected brand"
            )

        break_id = str(
            item["break_id"]
        )

        ad_break = ET.SubElement(
            root,
            qname("AdBreak"),
            {
                "timeOffset": (
                    format_time(
                        float(
                            item[
                                "timestamp_seconds"
                            ]
                        )
                    )
                ),
                "breakType": "linear",
                "breakId": break_id,
            },
        )

        ad_source = ET.SubElement(
            ad_break,
            qname("AdSource"),
            {
                "id": (
                    selected_brand[
                        "brand_id"
                    ]
                ),
                "allowMultipleAds": (
                    "false"
                ),
                "followRedirects": (
                    "true"
                ),
            },
        )

        vast_data = ET.SubElement(
            ad_source,
            qname("VASTAdData"),
        )

        vast = build_inline_vast(
            break_id=break_id,
            brand_id=(
                selected_brand[
                    "brand_id"
                ]
            ),
            brand_name=(
                selected_brand.get(
                    "name",
                    selected_brand[
                        "brand_id"
                    ],
                )
            ),
            creative_uri=(
                selected_brand[
                    "creative_uri"
                ]
            ),
            ad_duration_seconds=(
                ad_duration_seconds
            ),
        )

        vast_data.append(
            vast
        )

    return ET.ElementTree(
        root
    )


def write_vmap(
    *,
    path: Path,
    video_name: str,
    breaks: list[dict],
    ad_duration_seconds: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree = build_vmap(
        video_name=video_name,
        breaks=breaks,
        ad_duration_seconds=(
            ad_duration_seconds
        ),
    )

    ET.indent(
        tree,
        space="  ",
    )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )
