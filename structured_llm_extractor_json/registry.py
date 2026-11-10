"""
Single registry for document types: schema, label, few-shot, samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel

from schemas import (
    InvoiceSchema,
    ResumeSchema,
    PurchaseOrderSchema,
    JobDescriptionSchema,
    SoftValidationResult,
    validate_invoice_soft,
    validate_resume_soft,
    validate_po_soft,
    validate_jd_soft,
)


@dataclass(frozen=True)
class DocSpec:
    key: str
    label: str
    schema: Type[BaseModel]
    soft_validate: Callable[[BaseModel], SoftValidationResult]
    few_shot: str
    samples: tuple  # tuple of sample dicts


# Few-shots: multi-line style closer to real inputs; show missing fields as null
INVOICE_FEW_SHOT = """
### Example (clean invoice)
<document>
Invoice INV-1001
Date: 2024-06-01
From: Acme Supplies, 12 Oak St
To: Beta Retail
2x Widget @ 10.00 = 20.00
1x Cable @ 5.00 = 5.00
Subtotal 25.00
Tax 2.00
Total 27.00
Net 15
</document>

JSON:
{"invoice_number":"INV-1001","invoice_date":"2024-06-01","due_date":null,"currency":"USD","seller":{"name":"Acme Supplies","address":"12 Oak St","email":null,"phone":null,"tax_id":null},"buyer":{"name":"Beta Retail","address":null,"email":null,"phone":null,"tax_id":null},"line_items":[{"description":"Widget","quantity":2.0,"unit_price":10.0,"total":20.0,"tax_rate":null},{"description":"Cable","quantity":1.0,"unit_price":5.0,"total":5.0,"tax_rate":null}],"subtotal":25.0,"tax_amount":2.0,"discount":0.0,"total_amount":27.0,"payment_terms":"Net 15","notes":null,"status":null}
"""

RESUME_FEW_SHOT = """
### Example
<document>
Maria Lopez — Product Designer
maria@example.com | Berlin
Experience: Designer at Studio North (2020-2023) — led redesign of checkout.
Education: BA Design, RCA, 2019
Skills: Figma, Prototyping
</document>

JSON:
{"full_name":"Maria Lopez","email":"maria@example.com","phone":null,"location":"Berlin","linkedin":null,"github":null,"portfolio":null,"summary":null,"education":[{"institution":"RCA","degree":"BA","field_of_study":"Design","start_date":null,"end_date":"2019","gpa":null,"location":null}],"experience":[{"company":"Studio North","title":"Designer","location":null,"start_date":"2020","end_date":"2023","is_current":false,"description":null,"achievements":["led redesign of checkout"]}],"skills":[{"name":"Figma","category":null,"proficiency":null},{"name":"Prototyping","category":null,"proficiency":null}],"projects":[],"certifications":[],"languages":[]}
"""

PO_FEW_SHOT = """
### Example
<document>
PO-9001 / 2024-07-10
Buyer: City Library
Supplier: BookWorld Ltd
Items:
  10 Hardcover Atlas @ 12.00 = 120.00
  5 Map Set @ 8.00 = 40.00
Subtotal 160.00 Tax 0 (exempt) Total 160.00
</document>

JSON:
{"po_number":"PO-9001","po_date":"2024-07-10","delivery_date":null,"currency":"USD","buyer":{"name":"City Library","address":null,"email":null,"phone":null,"tax_id":null},"supplier":{"name":"BookWorld Ltd","address":null,"email":null,"phone":null,"tax_id":null},"ship_to":null,"line_items":[{"item_code":null,"description":"Hardcover Atlas","quantity":10.0,"unit":null,"unit_price":12.0,"total":120.0},{"item_code":null,"description":"Map Set","quantity":5.0,"unit":null,"unit_price":8.0,"total":40.0}],"subtotal":160.0,"tax_amount":0.0,"shipping_cost":0.0,"total_amount":160.0,"payment_terms":null,"shipping_method":null,"notes":null,"status":null}
"""

JD_FEW_SHOT = """
### Example
<document>
Data Analyst — GreenGrid Energy (Hybrid, Austin)
Full-time. $90k-$110k.
Own dashboards and reporting. Need SQL and Python. Benefits: health, 401k.
Apply: jobs.greengrid.example/da
</document>

JSON:
{"job_title":"Data Analyst","company":"GreenGrid Energy","location":"Hybrid, Austin","employment_type":"full-time","department":null,"salary_range":"$90k-$110k","salary_min":90000.0,"salary_max":110000.0,"salary_currency":"USD","salary_period":"year","posted_date":null,"application_deadline":null,"summary":null,"responsibilities":["Own dashboards and reporting"],"requirements":[{"category":"Required","items":["SQL","Python"]}],"benefits":["health","401k"],"skills":["SQL","Python"],"experience_years":null,"education_required":null,"report_to":null,"apply_url":"jobs.greengrid.example/da"}
"""


# Samples imported lazily to avoid circular imports at module load of synthetic data
def _build_registry() -> dict[str, DocSpec]:
    from synthetic_data import (
        SYNTHETIC_INVOICES,
        SYNTHETIC_RESUMES,
        SYNTHETIC_PURCHASE_ORDERS,
        SYNTHETIC_JOB_DESCRIPTIONS,
    )

    return {
        "invoice": DocSpec(
            key="invoice",
            label="Invoice",
            schema=InvoiceSchema,
            soft_validate=validate_invoice_soft,
            few_shot=INVOICE_FEW_SHOT,
            samples=tuple(SYNTHETIC_INVOICES),
        ),
        "resume": DocSpec(
            key="resume",
            label="Resume",
            schema=ResumeSchema,
            soft_validate=validate_resume_soft,
            few_shot=RESUME_FEW_SHOT,
            samples=tuple(SYNTHETIC_RESUMES),
        ),
        "purchase_order": DocSpec(
            key="purchase_order",
            label="Purchase Order",
            schema=PurchaseOrderSchema,
            soft_validate=validate_po_soft,
            few_shot=PO_FEW_SHOT,
            samples=tuple(SYNTHETIC_PURCHASE_ORDERS),
        ),
        "job_description": DocSpec(
            key="job_description",
            label="Job Description",
            schema=JobDescriptionSchema,
            soft_validate=validate_jd_soft,
            few_shot=JD_FEW_SHOT,
            samples=tuple(SYNTHETIC_JOB_DESCRIPTIONS),
        ),
    }


_REGISTRY: dict[str, DocSpec] | None = None


def get_registry() -> dict[str, DocSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_spec(doc_type: str) -> DocSpec:
    reg = get_registry()
    if doc_type not in reg:
        raise ValueError(f"Unknown doc type: {doc_type}")
    return reg[doc_type]


def list_doc_types() -> list[str]:
    return list(get_registry().keys())


def labels_map() -> dict[str, str]:
    return {k: v.label for k, v in get_registry().items()}
