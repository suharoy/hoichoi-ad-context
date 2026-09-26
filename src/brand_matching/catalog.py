"""Synthetic brand catalogue loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Brand:
    brand_id: str
    name: str
    description: str
    positive_contexts: tuple[str, ...]
    negative_contexts: tuple[str, ...]
    activities: tuple[str, ...]
    creative_uri: str


@dataclass(frozen=True)
class BrandCatalogue:
    catalogue_name: str
    catalogue_status: str
    brands: tuple[Brand, ...]


def load_catalogue(
    path: Path,
    *,
    allowed_contexts: set[str] | None = None,
) -> BrandCatalogue:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    brands = tuple(
        Brand(
            brand_id=str(
                item["brand_id"]
            ),
            name=str(
                item.get(
                    "name",
                    item["brand_id"],
                )
            ),
            description=str(
                item["description"]
            ),
            positive_contexts=tuple(
                str(value)
                for value
                in item.get(
                    "positive_contexts",
                    [],
                )
            ),
            negative_contexts=tuple(
                str(value)
                for value
                in item.get(
                    "negative_contexts",
                    [],
                )
            ),
            activities=tuple(
                str(value)
                for value
                in item.get(
                    "activities",
                    [],
                )
            ),
            creative_uri=str(
                item.get(
                    "creative_uri",
                    "",
                )
            ),
        )
        for item
        in payload["brands"]
    )

    ids = [
        brand.brand_id
        for brand in brands
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Brand catalogue contains duplicate brand_id values"
        )

    if allowed_contexts is not None:
        for brand in brands:
            referenced = (
                set(
                    brand.positive_contexts
                )
                | set(
                    brand.negative_contexts
                )
                | set(
                    brand.activities
                )
            )

            unknown = (
                referenced
                - allowed_contexts
            )

            if unknown:
                raise ValueError(
                    f"Brand {brand.brand_id} "
                    f"references unknown contexts: "
                    f"{sorted(unknown)}"
                )

    return BrandCatalogue(
        catalogue_name=str(
            payload["catalogue_name"]
        ),
        catalogue_status=str(
            payload[
                "catalogue_status"
            ]
        ),
        brands=brands,
    )
