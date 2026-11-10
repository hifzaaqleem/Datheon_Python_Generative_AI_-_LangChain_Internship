"""
Pydantic schemas for structured extraction.
Hard errors = structure/types. Soft warnings = arithmetic mismatches (returned to UI).
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Party(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = None


class SoftValidationResult(BaseModel):
    """Wrapper returned to the app: data + soft warnings."""
    data: dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


def _num_close(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(round(a, 2) - round(b, 2)) <= tol


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = Field(..., min_length=1)
    # Allow zero / negative for credit notes, returns, discount lines
    quantity: float = Field(...)
    unit_price: float = Field(...)
    total: float = Field(...)
    # Accept percent (8.25) or fraction (0.0825); store as percent when possible
    tax_rate: Optional[float] = None


class InvoiceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    # Accept "USD", "$", "US Dollar" — normalized later if possible
    currency: Optional[str] = None
    seller: Optional[Party] = None
    buyer: Optional[Party] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[Literal["paid", "unpaid", "overdue", "draft", "cancelled", "unknown"]] = None


def validate_invoice_soft(model: InvoiceSchema) -> SoftValidationResult:
    data = model.model_dump()
    warnings: list[str] = []

    if not model.invoice_number:
        warnings.append("Missing invoice_number.")
    if not model.invoice_date:
        warnings.append("Missing invoice_date.")
    if not model.seller or not model.seller.name:
        warnings.append("Missing seller name.")
    if not model.buyer or not model.buyer.name:
        warnings.append("Missing buyer name.")
    if not model.line_items:
        warnings.append("No line items extracted.")

    if model.line_items:
        calc_sub = round(sum(i.total for i in model.line_items), 2)
        if model.subtotal is not None and not _num_close(calc_sub, model.subtotal):
            warnings.append(
                f"Arithmetic: sum of line totals ({calc_sub}) ≠ subtotal ({model.subtotal}). "
                "Document may have a typo; values left as extracted."
            )
        for i, item in enumerate(model.line_items, 1):
            expected = round(item.quantity * item.unit_price, 2)
            if not _num_close(expected, item.total):
                warnings.append(
                    f"Arithmetic: line {i} ({item.description!r}): "
                    f"{item.quantity} × {item.unit_price} = {expected}, but total={item.total}."
                )

    if (
        model.subtotal is not None
        and model.total_amount is not None
    ):
        tax = model.tax_amount or 0.0
        disc = model.discount or 0.0
        expected_total = round(model.subtotal + tax - disc, 2)
        if not _num_close(expected_total, model.total_amount):
            warnings.append(
                f"Arithmetic: subtotal+tax-discount ({expected_total}) ≠ total_amount ({model.total_amount})."
            )

    if model.currency and len(model.currency.strip()) not in (1, 3, 4):
        warnings.append(f"Unusual currency value: {model.currency!r} (prefer ISO 3-letter code).")

    return SoftValidationResult(data=data, warnings=warnings)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class Education(BaseModel):
    model_config = ConfigDict(extra="ignore")

    institution: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # No upper bound — scales differ (4.0, 5.0, 10.0, percentages)
    gpa: Optional[float] = None
    location: Optional[str] = None


class Experience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)


class Skill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    category: Optional[str] = None
    proficiency: Optional[str] = None


class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ResumeSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    summary: Optional[str] = None
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


def validate_resume_soft(model: ResumeSchema) -> SoftValidationResult:
    data = model.model_dump()
    warnings: list[str] = []
    if not model.full_name:
        warnings.append("Missing full_name.")
    if not model.experience and not model.education:
        warnings.append("No experience or education extracted.")
    return SoftValidationResult(data=data, warnings=warnings)


# ---------------------------------------------------------------------------
# Purchase Order
# ---------------------------------------------------------------------------

class POLineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_code: Optional[str] = None
    description: str = Field(..., min_length=1)
    quantity: float = Field(...)
    unit: Optional[str] = None
    unit_price: float = Field(...)
    total: float = Field(...)


class PurchaseOrderSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    po_number: Optional[str] = None
    po_date: Optional[str] = None
    delivery_date: Optional[str] = None
    currency: Optional[str] = None
    buyer: Optional[Party] = None
    supplier: Optional[Party] = None
    ship_to: Optional[str] = None
    line_items: List[POLineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    shipping_cost: Optional[float] = None
    total_amount: Optional[float] = None
    payment_terms: Optional[str] = None
    shipping_method: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None


def validate_po_soft(model: PurchaseOrderSchema) -> SoftValidationResult:
    data = model.model_dump()
    warnings: list[str] = []
    if not model.po_number:
        warnings.append("Missing po_number.")
    if not model.line_items:
        warnings.append("No line items extracted.")
    if model.line_items:
        calc = round(sum(i.total for i in model.line_items), 2)
        if model.subtotal is not None and not _num_close(calc, model.subtotal):
            warnings.append(
                f"Arithmetic: sum of line totals ({calc}) ≠ subtotal ({model.subtotal})."
            )
        for i, item in enumerate(model.line_items, 1):
            expected = round(item.quantity * item.unit_price, 2)
            if not _num_close(expected, item.total):
                warnings.append(
                    f"Arithmetic: line {i} ({item.description!r}): "
                    f"{item.quantity} × {item.unit_price} = {expected}, total={item.total}."
                )
    if model.subtotal is not None and model.total_amount is not None:
        tax = model.tax_amount or 0.0
        ship = model.shipping_cost or 0.0
        expected = round(model.subtotal + tax + ship, 2)
        if not _num_close(expected, model.total_amount):
            warnings.append(
                f"Arithmetic: subtotal+tax+shipping ({expected}) ≠ total_amount ({model.total_amount})."
            )
    return SoftValidationResult(data=data, warnings=warnings)


# ---------------------------------------------------------------------------
# Job Description
# ---------------------------------------------------------------------------

class JobRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str = Field(..., description="Required, Preferred, Nice-to-have, etc.")
    items: List[str] = Field(default_factory=list)


class JobDescriptionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[
        Literal["full-time", "part-time", "contract", "internship", "temporary", "remote", "hybrid", "other"]
    ] = None
    department: Optional[str] = None
    # Prefer structured salary when possible
    salary_range: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[Literal["year", "month", "hour", "week", "day"]] = None
    posted_date: Optional[str] = None
    application_deadline: Optional[str] = None
    summary: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[JobRequirement] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[str] = None
    education_required: Optional[str] = None
    report_to: Optional[str] = None
    apply_url: Optional[str] = None


def validate_jd_soft(model: JobDescriptionSchema) -> SoftValidationResult:
    data = model.model_dump()
    warnings: list[str] = []
    if not model.job_title:
        warnings.append("Missing job_title.")
    if not model.company:
        warnings.append("Missing company.")
    return SoftValidationResult(data=data, warnings=warnings)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

SCHEMA_BY_TYPE = {
    "invoice": InvoiceSchema,
    "resume": ResumeSchema,
    "purchase_order": PurchaseOrderSchema,
    "job_description": JobDescriptionSchema,
}

SOFT_VALIDATORS = {
    "invoice": validate_invoice_soft,
    "resume": validate_resume_soft,
    "purchase_order": validate_po_soft,
    "job_description": validate_jd_soft,
}

DOC_TYPE_LABELS = {
    "invoice": "Invoice",
    "resume": "Resume",
    "purchase_order": "Purchase Order",
    "job_description": "Job Description",
}


def get_schema_for_type(doc_type: str):
    if doc_type not in SCHEMA_BY_TYPE:
        raise ValueError(f"Unsupported type: {doc_type}. Choose from {list(SCHEMA_BY_TYPE)}")
    return SCHEMA_BY_TYPE[doc_type]


def soft_validate(doc_type: str, model: BaseModel) -> SoftValidationResult:
    return SOFT_VALIDATORS[doc_type](model)
