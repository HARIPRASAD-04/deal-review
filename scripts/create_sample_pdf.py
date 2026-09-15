"""Script to generate the synthetic deal PDF fixture for Module 2 tests.

Run from the project root::

    python scripts/create_sample_pdf.py

Produces: data/samples/deal_002.pdf

The PDF is a realistic but completely synthetic 3-page commercial loan term
sheet.  It contains:
    - Page 1: Cover page, parties, facility summary
    - Page 2: Financial terms, conditions precedent, covenants
    - Page 3: Events of default, representations & warranties, execution

The document includes multiple clearly identifiable sections/clauses,
monetary amounts, percentages, dates, legal qualifiers, and obligations —
all of which can later be extracted deterministically by the Evidence Registry.
"""

from __future__ import annotations

import pathlib
import sys

import pymupdf as fitz  # PyMuPDF (pymupdf package; fitz is the legacy alias)


OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "data" / "samples" / "deal_002.pdf"

PAGES: list[list[tuple[float, float, float, str]]] = [
    # Page 1: (x, y, font_size, text)
    [
        (72, 60, 16, "COMMERCIAL LOAN TERM SHEET"),
        (72, 90, 12, "DEAL-002  |  CONFIDENTIAL  |  DRAFT"),
        (72, 130, 11, "1. PARTIES"),
        (72, 150, 10, "Borrower:  Apex Manufacturing Pvt. Ltd.\n"
                      "           (CIN: U28910MH2005PTC154321)\n"
                      "           Registered Office: 42 Industrial Estate, Pune, Maharashtra 411019"),
        (72, 215, 10, "Lender:    Horizon Commercial Bank Ltd.\n"
                      "           (Reg. No. 0004521)\n"
                      "           Head Office: 9 Banking Square, Mumbai, Maharashtra 400001"),
        (72, 275, 11, "2. FACILITY DETAILS"),
        (72, 295, 10, "Type of Facility:  Term Loan"),
        (72, 315, 10, "Principal Amount:  INR 50,000,000 (Indian Rupees Five Crore only)"),
        (72, 335, 10, "Purpose:           To finance the acquisition of manufacturing equipment\n"
                      "                   and for working capital requirements as detailed in\n"
                      "                   Schedule I of this Term Sheet."),
        (72, 390, 10, "Tenure:            60 months (5 years) from the date of first disbursement."),
        (72, 410, 10, "Moratorium:        6 months on principal repayment from the date of\n"
                      "                   first disbursement; interest to be serviced monthly."),
        (72, 455, 11, "3. SECURITY"),
        (72, 475, 10, "Primary Security:  First-ranking exclusive charge over the machinery\n"
                      "                   and equipment purchased out of the facility proceeds,\n"
                      "                   valued at not less than INR 70,000,000."),
        (72, 530, 10, "Collateral:        Personal guarantee of Mr. Rajesh Kumar (DIN: 00123456),\n"
                      "                   Director of the Borrower, together with a registered\n"
                      "                   equitable mortgage over immovable property located at\n"
                      "                   Survey No. 14/2, Village Chakan, Pune, Maharashtra,\n"
                      "                   valued at INR 35,000,000, subject to title verification."),
        (72, 620, 10, "Date:  30 June 2024"),
        (72, 640, 10, "Reference:  HCB/TL/2024/002"),
    ],
    # Page 2
    [
        (72, 60, 11, "4. FINANCIAL TERMS"),
        (72, 80, 10, "4.1 - Interest Rate\n"
                     "The Loan shall carry interest at the rate of 8.5% per annum (p.a.),\n"
                     "computed on a monthly reducing balance basis.  The interest rate is\n"
                     "subject to revision by the Lender at quarterly intervals, provided that\n"
                     "the Lender shall give not less than 30 (thirty) days' prior written\n"
                     "notice of any upward revision exceeding 50 basis points."),
        (72, 175, 10, "4.2 - Processing Fee\n"
                      "A non-refundable processing fee of 1.00% of the sanctioned facility\n"
                      "amount (i.e., INR 500,000) shall be payable by the Borrower at the\n"
                      "time of execution of this Term Sheet.  GST and other applicable\n"
                      "taxes shall be charged additionally."),
        (72, 255, 10, "4.3 - Prepayment\n"
                      "The Borrower may prepay the outstanding loan amount, in full or in\n"
                      "part, subject to a prepayment premium of 2% of the amount prepaid,\n"
                      "except where such prepayment is made from internal accruals of the\n"
                      "Borrower, in which case no premium shall be charged, provided that\n"
                      "the Borrower provides at least 15 (fifteen) days' advance notice."),
        (72, 355, 11, "5. CONDITIONS PRECEDENT"),
        (72, 375, 10, "5.1 - Documentation\n"
                      "Disbursement is subject to completion of all documentation to the\n"
                      "satisfaction of the Lender, including but not limited to:\n"
                      "(a) Executed Loan Agreement and Guarantee Deed;\n"
                      "(b) Registered mortgage documentation;\n"
                      "(c) Board Resolution authorising the borrowing;\n"
                      "(d) KYC documents for the Borrower and Guarantor."),
        (72, 480, 10, "5.2 - Financial Covenants (Ongoing)\n"
                      "The Borrower shall at all times maintain:\n"
                      "(a) A minimum Debt Service Coverage Ratio (DSCR) of 1.25x, measured\n"
                      "    on a trailing 12-month basis as at the end of each financial year;\n"
                      "(b) A Current Ratio of not less than 1.10x;\n"
                      "(c) Total outside liabilities to tangible net worth ratio not exceeding\n"
                      "    3.0x."),
        (72, 600, 10, "5.3 - Insurance\n"
                      "The Borrower shall maintain comprehensive insurance over all primary\n"
                      "security assets at replacement value, with the Lender noted as the\n"
                      "sole loss payee.  Proof of insurance shall be submitted to the Lender\n"
                      "within 30 days of each renewal."),
    ],
    # Page 3
    [
        (72, 60, 11, "6. EVENTS OF DEFAULT"),
        (72, 80, 10, "Each of the following shall constitute an Event of Default:\n"
                     "(a) Non-payment of any instalment of principal or interest within\n"
                     "    7 (seven) days of the due date;\n"
                     "(b) Breach of any financial covenant set out in Section 5.2, unless\n"
                     "    remedied within 30 days of written notice from the Lender;\n"
                     "(c) Any material adverse change in the financial condition of the\n"
                     "    Borrower or Guarantor, as determined by the Lender in its sole\n"
                     "    discretion;\n"
                     "(d) Insolvency, liquidation, or dissolution of the Borrower;\n"
                     "(e) Any representation or warranty made herein proves materially\n"
                     "    incorrect as of the date when made or deemed repeated."),
        (72, 235, 11, "7. REPRESENTATIONS AND WARRANTIES"),
        (72, 255, 10, "7.1\n"
                      "The Borrower represents and warrants that:\n"
                      "(a) It is duly incorporated, validly existing, and in good standing\n"
                      "    under the laws of India;\n"
                      "(b) The execution of this Term Sheet and the Loan Agreement has been\n"
                      "    duly authorised by its Board of Directors;\n"
                      "(c) There is no pending or threatened litigation that would materially\n"
                      "    and adversely affect its ability to repay the Facility;\n"
                      "(d) All financial statements provided to the Lender are true, fair,\n"
                      "    and accurate in all material respects as at their respective dates."),
        (72, 400, 11, "8. GOVERNING LAW AND JURISDICTION"),
        (72, 420, 10, "This Term Sheet and all disputes arising out of or in connection with\n"
                      "it shall be governed by the laws of India.  The parties submit to the\n"
                      "exclusive jurisdiction of the courts at Mumbai, Maharashtra."),
        (72, 480, 11, "9. ACCEPTANCE"),
        (72, 500, 10, "This Term Sheet is valid for a period of 30 (thirty) days from the\n"
                      "date of issue.  Acceptance is to be indicated by the authorised\n"
                      "signatory of the Borrower returning a duly signed copy to the Lender."),
        (72, 575, 10, "FOR HORIZON COMMERCIAL BANK LTD."),
        (72, 600, 10, "Authorised Signatory\nName: ____________________________\nDesignation: ____________________________"),
        (72, 655, 10, "FOR APEX MANUFACTURING PVT. LTD."),
        (72, 680, 10, "Authorised Signatory\nName: ____________________________\nDesignation: ____________________________"),
        (72, 730, 9, "Page 3 of 3  |  DEAL-002  |  DRAFT - NOT FOR CIRCULATION"),
    ],
]


def create_pdf(output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    for page_content in PAGES:
        page = doc.new_page(width=595, height=842)  # A4
        for x, y, font_size, text in page_content:
            page.insert_text(
                (x, y),
                text,
                fontsize=font_size,
                fontname="helv",
                color=(0, 0, 0),
            )

    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}  ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_pdf(OUTPUT_PATH)
    sys.exit(0)
