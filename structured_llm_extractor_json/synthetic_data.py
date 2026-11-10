"""
Synthetic samples. Includes clean demos AND messy / edge cases.
Some entries include expected_json for evaluation.
"""

# ---------------------------------------------------------------------------
# Invoices
# ---------------------------------------------------------------------------

SYNTHETIC_INVOICES = [
    {
        "id": "inv_001",
        "title": "Clean product invoice",
        "messy": False,
        "text": """
        INVOICE INV-88421
        Date: 2024-09-12  Due: 2024-10-12
        From: BrightTech Solutions LLC, 789 Innovation Drive, Austin TX
        Email: billing@brighttech.io  Tax ID: 87-6543210
        Bill To: Horizon Retail Group, 321 Market Street, Seattle WA

        Wireless Mouse Pro     25  @ 29.99  = 749.75
        USB-C Hub 7-in-1       10  @ 49.50  = 495.00
        Laptop Stand            8  @ 34.00  = 272.00

        Subtotal 1516.75
        Tax 8.25% = 125.13
        Discount -50.00
        TOTAL 1591.88
        Terms: Net 30
        """,
    },
    {
        "id": "inv_messy_001",
        "title": "Messy: OCR noise + math mismatch",
        "messy": True,
        "text": """
        1NVOICE # 1NV-9O21  Date 12/0ct/24
        Seller: N0rthwind Trad1ng
        Client: acme co (attn AP)

        widgets x3  @ 1O.OO  total 3O.OO
        service call 1 @ 50 = 55.00   << total does not match unit price

        subtotl: 85
        tax: ??
        TOTAL DUE: 90.00

        pay when you can
        """,
        "expected_json": {
            "invoice_number": "1NV-9O21",
            "invoice_date": "12/0ct/24",
            "seller": {"name": "N0rthwind Trad1ng"},
            "buyer": {"name": "acme co"},
            "line_items": [
                {"description": "widgets", "quantity": 3.0, "unit_price": 10.0, "total": 30.0},
                {"description": "service call", "quantity": 1.0, "unit_price": 50.0, "total": 55.0},
            ],
            "subtotal": 85.0,
            "total_amount": 90.0,
        },
    },
    {
        "id": "inv_credit_001",
        "title": "Credit note (negative line)",
        "messy": True,
        "text": """
        CREDIT NOTE CN-220
        Date: 2024-11-01
        From: SpareParts Inc
        To: Garage LLC
        Return: Brake pads qty 2 @ 15.00 = -30.00
        Restocking fee 5.00
        Credit total: -25.00
        """,
    },
]

# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------

SYNTHETIC_RESUMES = [
    {
        "id": "res_001",
        "title": "Clean data scientist",
        "messy": False,
        "text": """
        SARAH CHEN | Data Scientist
        sarah.chen@email.com | San Francisco
        Summary: 7 years production ML.
        Experience:
        Staff DS @ Stripe (2022-Present) — fraud models
        Senior DS @ Airbnb (2019-2022)
        Education: PhD CS Stanford 2017; BS Stats Michigan 2013
        Skills: Python, PyTorch, AWS
        """,
    },
    {
        "id": "res_messy_001",
        "title": "Messy: sparse + non-US GPA scale",
        "messy": True,
        "text": """
        ravi patel
        mumbai | ravip@mail.in

        worked at tata consultancy 2018-2021 software eng
        then freelancing

        B.Tech IT VJTI mumbai 2018 GPA 8.7/10

        java sql docker
        """,
        "expected_json": {
            "full_name": "ravi patel",
            "email": "ravip@mail.in",
            "location": "mumbai",
            "education": [
                {
                    "institution": "VJTI mumbai",
                    "degree": "B.Tech",
                    "field_of_study": "IT",
                    "end_date": "2018",
                    "gpa": 8.7,
                }
            ],
            "experience": [
                {
                    "company": "tata consultancy",
                    "title": "software eng",
                    "start_date": "2018",
                    "end_date": "2021",
                    "is_current": False,
                }
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Purchase orders
# ---------------------------------------------------------------------------

SYNTHETIC_PURCHASE_ORDERS = [
    {
        "id": "po_001",
        "title": "Clean office equipment PO",
        "messy": False,
        "text": """
        PURCHASE ORDER PO-2024-7781
        Date: 2024-10-08  Delivery: 2024-10-25
        Buyer: NovaSoft Inc., 400 Innovation Parkway, Boston MA
        Supplier: DeskPro Furniture Co., Grand Rapids MI
        Ship To: NovaSoft HQ – Receiving Dock B

        DP-STAND-01  Adjustable Standing Desk  12 pcs  389.00  4668.00
        DP-CHAIR-X   Ergonomic Mesh Chair      12 pcs  245.00  2940.00
        DP-MON-27    27" 4K Monitor            24 pcs  329.00  7896.00
        DP-CABLE-KIT Cable Management Kit      12 set   45.00   540.00

        Subtotal 16044.00  Tax 1002.75  Shipping 320.00  TOTAL 17366.75
        Net 30 | LTL Freight
        """,
    },
    {
        "id": "po_messy_001",
        "title": "Messy: missing totals + mixed units",
        "messy": True,
        "text": """
        po ref 44-b buyer warehouse 3 supplier unknown vendor
        need by friday

        100 kg rice ?? price not listed
        2 boxes tape @ 3.5 each

        please confirm
        """,
    },
]

# ---------------------------------------------------------------------------
# Job descriptions
# ---------------------------------------------------------------------------

SYNTHETIC_JOB_DESCRIPTIONS = [
    {
        "id": "jd_001",
        "title": "Clean product marketing role",
        "messy": False,
        "text": """
        Product Marketing Manager – B2B SaaS
        Company: PulseMetrics
        Location: Hybrid – New York, NY
        Type: Full-time
        Compensation: $120k–$145k base + bonus

        Own positioning and GTM for analytics product line.
        Responsibilities: messaging frameworks; launch plans; competitive analysis.
        Required: 4+ years PMM B2B SaaS; strong writing.
        Nice-to-have: analytics or developer-tools background.
        Perks: Unlimited PTO, 401(k) match.
        Reports to: Head of Marketing
        Apply: jobs@pulsemetrics.com
        """,
    },
    {
        "id": "jd_messy_001",
        "title": "Messy: minimal posting",
        "messy": True,
        "text": """
        hiring cook for night shift downtown
        cash ok experience preferred
        text 555-0199
        """,
    },
]


def get_synthetic_samples(doc_type: str, include_messy: bool = True):
    mapping = {
        "invoice": SYNTHETIC_INVOICES,
        "resume": SYNTHETIC_RESUMES,
        "purchase_order": SYNTHETIC_PURCHASE_ORDERS,
        "job_description": SYNTHETIC_JOB_DESCRIPTIONS,
    }
    samples = mapping.get(doc_type, [])
    if not include_messy:
        samples = [s for s in samples if not s.get("messy")]
    return samples
