"""
ChronoVeritas v2 — Task Bank
Seed facts and document templates used by Mutator + Spreader to generate training tasks.
All claims are fictional but realistic. Domain coverage: municipal, corporate, scientific.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random


@dataclass
class SeedFact:
    """A single verifiable true claim with its primary source document."""
    fact_id: str
    domain: str                    # "municipal" | "corporate" | "scientific"
    true_claim: str                # The ground-truth accurate statement
    true_number: Optional[str]     # Primary number/stat that could be distorted
    true_entity: str               # The entity the claim is about
    true_date: str                 # When this event occurred
    primary_source: Dict           # The Tier-1 authoritative document
    secondary_sources: List[Dict]  # Tier-2 documents (corroborating, pre-mutation)


# ── Seed facts across 3 domains, 5 per domain ─────────────────────────────

SEED_FACTS: List[SeedFact] = [

    # ── MUNICIPAL domain ──────────────────────────────────────────────────

    SeedFact(
        fact_id="MUN-001",
        domain="municipal",
        true_claim="The Riverdale City Council approved a 5% increase to the public transit budget for fiscal year 2024.",
        true_number="5",
        true_entity="Riverdale City Council",
        true_date="2024-01-15",
        primary_source={
            "title": "Riverdale City Council — Meeting Minutes, January 2024",
            "source": "Riverdale City Clerk Office",
            "reliability_tier": 1,
            "content": (
                "RIVERDALE CITY COUNCIL — OFFICIAL MEETING MINUTES\n"
                "Date: January 15, 2024 | Session: Regular Council Meeting\n\n"
                "RESOLUTION 2024-07: PUBLIC TRANSIT BUDGET AMENDMENT\n"
                "Moved by Councilmember Hargrove, seconded by Councilmember Chen.\n"
                "Following review of ridership data and operational costs, the Council "
                "unanimously approves a 5% increase to the Department of Public Transit "
                "annual budget, raising the allocation from $40.2M to $42.2M for FY2024. "
                "The increase is designated for fleet maintenance and driver retention.\n"
                "VOTE: 7-0 in favour. Resolution adopted."
            ),
        },
        secondary_sources=[
            {
                "title": "City Budget Office — FY2024 Appropriations Summary",
                "source": "Riverdale Budget Office",
                "reliability_tier": 1,
                "content": (
                    "FY2024 DEPARTMENT APPROPRIATIONS — FINAL SUMMARY\n"
                    "Department of Public Transit: $42,210,000\n"
                    "Prior Year: $40,200,000 | Change: +$2,010,000 (+5.0%)\n"
                    "Purpose: Fleet maintenance ($1.2M) and driver retention ($0.81M).\n"
                    "Approved: City Council Resolution 2024-07, January 15, 2024."
                ),
            },
        ],
    ),

    SeedFact(
        fact_id="MUN-002",
        domain="municipal",
        true_claim="Westfield County commissioners approved a 12-month pilot program for free senior bus passes, effective March 2024.",
        true_number="12",
        true_entity="Westfield County",
        true_date="2024-02-08",
        primary_source={
            "title": "Westfield County Board of Commissioners — Resolution 2024-14",
            "source": "Westfield County Clerk",
            "reliability_tier": 1,
            "content": (
                "WESTFIELD COUNTY BOARD OF COMMISSIONERS\n"
                "Resolution 2024-14 | Adopted: February 8, 2024\n\n"
                "SUBJECT: Senior Mobility Pilot Program\n\n"
                "RESOLVED: The Board authorises a 12-month pilot program providing "
                "zero-cost transit passes to county residents aged 65 and older, "
                "commencing March 1, 2024 and concluding February 28, 2025. "
                "Estimated program cost: $340,000, funded from the Community Services Reserve Fund.\n"
                "Commissioner votes: 4-1 in favour."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="MUN-003",
        domain="municipal",
        true_claim="The Northgate school district reported a 3.2% decline in enrollment for the 2023-2024 academic year, totalling 18,450 students.",
        true_number="3.2",
        true_entity="Northgate Unified School District",
        true_date="2023-09-15",
        primary_source={
            "title": "Northgate USD — Annual Enrollment Report 2023-24",
            "source": "Northgate Unified School District",
            "reliability_tier": 1,
            "content": (
                "NORTHGATE UNIFIED SCHOOL DISTRICT\n"
                "Annual Enrollment Statistical Report — Academic Year 2023-24\n\n"
                "Total District Enrollment (October count): 18,450 students\n"
                "Prior Year Enrollment (2022-23): 19,057 students\n"
                "Year-over-Year Change: -607 students (-3.19%)\n\n"
                "Enrollment has declined for the third consecutive year, consistent "
                "with county-wide demographic trends. The District projects stabilisation "
                "by 2025-26 based on current birth-rate data.\n"
                "Report prepared by: Office of Assessment and Accountability, October 1, 2023."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="MUN-004",
        domain="municipal",
        true_claim="Crestview municipality awarded a $4.7M infrastructure contract to Hartwell Construction for road resurfacing, completing work by Q4 2024.",
        true_number="4.7",
        true_entity="Crestview",
        true_date="2024-03-22",
        primary_source={
            "title": "Crestview Procurement Notice — Contract Award 2024-INF-009",
            "source": "Crestview Department of Public Works",
            "reliability_tier": 1,
            "content": (
                "OFFICIAL CONTRACT AWARD NOTICE\n"
                "Contract ID: 2024-INF-009 | Awarded: March 22, 2024\n\n"
                "Scope: Arterial road resurfacing — 23.4 lane-miles across 6 corridors\n"
                "Awardee: Hartwell Construction Group, Inc.\n"
                "Contract Value: $4,712,400 (fixed price)\n"
                "Completion Deadline: December 15, 2024\n\n"
                "Selection basis: Lowest responsive bid. Three bids received; "
                "Hartwell bid $4.71M vs. next-lowest $5.3M.\n"
                "Approved by: Director of Public Works, March 22, 2024."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="MUN-005",
        domain="municipal",
        true_claim="Lakeview City reduced property tax rates by 0.8% for the 2024 fiscal year, the first reduction in seven years.",
        true_number="0.8",
        true_entity="Lakeview City",
        true_date="2023-12-05",
        primary_source={
            "title": "Lakeview City Council — Tax Rate Ordinance 2023-41",
            "source": "Lakeview City Finance Department",
            "reliability_tier": 1,
            "content": (
                "LAKEVIEW CITY — PROPERTY TAX RATE ORDINANCE 2023-41\n"
                "Adopted: December 5, 2023 | Effective: January 1, 2024\n\n"
                "The City Council hereby sets the FY2024 general property tax levy rate "
                "at 6.92 mills per $1,000 assessed value, representing a reduction of "
                "0.8% from the prior year rate of 6.975 mills.\n\n"
                "This is the first reduction in the municipal tax rate since FY2017. "
                "The reduction reflects a $2.1M surplus carried forward from FY2023.\n"
                "Adopted unanimously (8-0)."
            ),
        },
        secondary_sources=[],
    ),

    # ── CORPORATE domain ──────────────────────────────────────────────────

    SeedFact(
        fact_id="CORP-001",
        domain="corporate",
        true_claim="GlobalTech Corp transferred 800 employees to a new subsidiary and 400 accepted voluntary separation packages in Q3 2023.",
        true_number="800",
        true_entity="GlobalTech Corp",
        true_date="2023-10-30",
        primary_source={
            "title": "GlobalTech Corp — SEC Form 10-Q (Q3 2023)",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "GLOBALTECH CORP — QUARTERLY REPORT (FORM 10-Q)\n"
                "Period: Quarter ended September 30, 2023\n"
                "Filed: October 30, 2023\n\n"
                "NOTE 7 — WORKFORCE RESTRUCTURING\n"
                "In Q3 2023, the Company completed a restructuring of its legacy "
                "Infrastructure Services division. A total of 800 employees were "
                "transferred to GlobalTech Infrastructure LLC, a newly formed wholly-owned "
                "subsidiary. An additional 400 employees elected voluntary separation "
                "under the Company's Early Transition Program, receiving severance packages "
                "averaging 18 weeks of compensation.\n\n"
                "Revenue for Q3 2023: $1.87B (+3.1% year-over-year).\n"
                "No involuntary terminations occurred in connection with this restructuring."
            ),
        },
        secondary_sources=[
            {
                "title": "GlobalTech Corp — Q3 2023 Earnings Press Release",
                "source": "GlobalTech Investor Relations",
                "reliability_tier": 2,
                "content": (
                    "GLOBALTECH REPORTS THIRD QUARTER 2023 RESULTS\n"
                    "Q3 Revenue: $1.87 billion, up 3.1% year-over-year\n\n"
                    "The Company completed the separation of its Infrastructure Services "
                    "division in Q3. 800 employees joined the new subsidiary GlobalTech "
                    "Infrastructure LLC, while 400 team members chose early transition "
                    "packages. The Company reaffirms its FY2023 revenue guidance of $7.4-7.6B."
                ),
            },
        ],
    ),

    SeedFact(
        fact_id="CORP-002",
        domain="corporate",
        true_claim="Meridian Pharma reported a 14% increase in net income for fiscal 2023, driven by its oncology drug portfolio.",
        true_number="14",
        true_entity="Meridian Pharma",
        true_date="2024-02-15",
        primary_source={
            "title": "Meridian Pharma — Annual Report (Form 10-K FY2023)",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "MERIDIAN PHARMACEUTICAL CORP — ANNUAL REPORT (FORM 10-K)\n"
                "Fiscal Year Ended: December 31, 2023 | Filed: February 15, 2024\n\n"
                "FINANCIAL HIGHLIGHTS\n"
                "Net income FY2023: $2.34 billion\n"
                "Net income FY2022: $2.05 billion\n"
                "Year-over-year increase: +13.9% (~14%)\n\n"
                "Growth was driven primarily by the oncology portfolio, particularly "
                "Meridian's flagship drug ONCOREL, which achieved $890M in global sales "
                "(up 22% vs. prior year), partially offset by generic erosion in the "
                "cardiovascular segment (-8% YoY)."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="CORP-003",
        domain="corporate",
        true_claim="Apex Logistics reduced its carbon emissions by 18% in 2023 compared to its 2019 baseline, ahead of its 2025 target.",
        true_number="18",
        true_entity="Apex Logistics",
        true_date="2024-03-01",
        primary_source={
            "title": "Apex Logistics — 2023 ESG and Sustainability Report",
            "source": "Apex Logistics Corp",
            "reliability_tier": 1,
            "content": (
                "APEX LOGISTICS — 2023 ENVIRONMENTAL, SOCIAL, AND GOVERNANCE REPORT\n"
                "Published: March 1, 2024 | Verified by: Deloitte Sustainability Assurance\n\n"
                "CLIMATE PERFORMANCE\n"
                "Scope 1+2 greenhouse gas emissions (2023): 412,000 metric tons CO2e\n"
                "Baseline (2019): 502,000 metric tons CO2e\n"
                "Reduction: -90,000 metric tons (-17.9%, approximately 18%)\n\n"
                "This performance exceeds our 2025 target of 15% reduction by 3 percentage "
                "points, two years ahead of schedule. Emissions reduction was achieved through "
                "fleet electrification (23% of delivery vehicles now EV) and renewable energy "
                "procurement for all owned facilities."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="CORP-004",
        domain="corporate",
        true_claim="Vantage Retail closed 47 underperforming store locations in Q1 2024, affecting approximately 1,100 employees.",
        true_number="47",
        true_entity="Vantage Retail",
        true_date="2024-04-22",
        primary_source={
            "title": "Vantage Retail Inc. — Q1 2024 10-Q Filing",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "VANTAGE RETAIL INC — QUARTERLY REPORT (FORM 10-Q)\n"
                "Quarter Ended: March 31, 2024 | Filed: April 22, 2024\n\n"
                "STORE PORTFOLIO\n"
                "During Q1 2024, the Company closed 47 store locations identified in the "
                "FY2023 annual strategic review as underperforming against return-on-investment "
                "thresholds. Affected workforce: approximately 1,100 employees. "
                "Affected employees were offered severance and preferential placement at "
                "remaining locations where vacancies exist. The Company operates 1,804 "
                "locations following these closures (down from 1,851 at year-end 2023)."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="CORP-005",
        domain="corporate",
        true_claim="Citadel Bank's non-performing loan ratio improved from 2.8% to 1.9% in FY2023, below the industry average of 2.4%.",
        true_number="1.9",
        true_entity="Citadel Bank",
        true_date="2024-01-31",
        primary_source={
            "title": "Citadel Bancorp — Annual Report FY2023",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "CITADEL BANCORP — ANNUAL REPORT (FORM 10-K)\n"
                "Fiscal Year Ended: December 31, 2023 | Filed: January 31, 2024\n\n"
                "CREDIT QUALITY\n"
                "Non-performing loan (NPL) ratio as of December 31, 2023: 1.9%\n"
                "NPL ratio as of December 31, 2022: 2.8%\n"
                "Improvement: -90 basis points year-over-year\n\n"
                "The FY2023 NPL ratio of 1.9% compares favourably to the FDIC-reported "
                "peer average of 2.4% for community and mid-size banks. The improvement "
                "reflects enhanced underwriting standards implemented in 2021 and "
                "accelerated workout of legacy commercial real estate exposures."
            ),
        },
        secondary_sources=[],
    ),

    # ── SCIENTIFIC domain ─────────────────────────────────────────────────

    SeedFact(
        fact_id="SCI-001",
        domain="scientific",
        true_claim="A peer-reviewed study found that the drug Neovex reduced tumour size by 34% in Phase II trials, with 12% of participants experiencing serious adverse events.",
        true_number="34",
        true_entity="Neovex",
        true_date="2023-11-08",
        primary_source={
            "title": "Phase II Clinical Trial Results: Neovex in Advanced Colorectal Cancer",
            "source": "Journal of Clinical Oncology",
            "reliability_tier": 1,
            "content": (
                "ORIGINAL ARTICLE — Journal of Clinical Oncology\n"
                "Published: November 8, 2023 | DOI: 10.1200/JCO.23.01847\n\n"
                "ABSTRACT\n"
                "Background: Neovex (compound NVX-440) is a novel KRAS inhibitor.\n"
                "Methods: Randomised Phase II trial, n=184 patients with advanced colorectal "
                "cancer (KRAS G12C mutation). Primary endpoint: objective response rate.\n\n"
                "Results: Neovex demonstrated a mean tumour size reduction of 34% "
                "(95% CI: 28.4–39.6%) in the intent-to-treat population. "
                "Serious adverse events (Grade ≥3) occurred in 12% of participants (n=22), "
                "primarily hepatotoxicity and fatigue. No treatment-related deaths were observed.\n\n"
                "Conclusions: Neovex shows clinically meaningful activity. Phase III "
                "investigation is warranted."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-002",
        domain="scientific",
        true_claim="Researchers at Caldera University found that daily 30-minute walking reduced systolic blood pressure by an average of 6 mmHg in adults with hypertension over 12 weeks.",
        true_number="6",
        true_entity="Caldera University",
        true_date="2024-01-22",
        primary_source={
            "title": "Walking Intervention and Blood Pressure: A Randomised Controlled Trial",
            "source": "American Heart Journal",
            "reliability_tier": 1,
            "content": (
                "RANDOMISED CONTROLLED TRIAL — American Heart Journal\n"
                "Published: January 22, 2024\n\n"
                "STUDY DESIGN: 12-week RCT; n=210 adults with Stage 1 hypertension "
                "(SBP 130-159 mmHg). Intervention: 30 min brisk walking daily. "
                "Control: usual care.\n\n"
                "PRIMARY OUTCOME: Change in systolic blood pressure (SBP) at 12 weeks.\n"
                "Results: Intervention group mean SBP reduction: -6.2 mmHg (SD ±3.1)\n"
                "Control group mean SBP change: -1.1 mmHg (SD ±2.8)\n"
                "Between-group difference: -5.1 mmHg (p<0.001)\n\n"
                "Caldera University, Department of Preventive Medicine."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-003",
        domain="scientific",
        true_claim="A NASA-funded study estimated the Helios-4 asteroid has a 0.003% probability of Earth impact within the next 100 years.",
        true_number="0.003",
        true_entity="Helios-4 asteroid",
        true_date="2024-02-29",
        primary_source={
            "title": "Orbital Mechanics Analysis: Near-Earth Object 2024 HX4 (Helios-4)",
            "source": "NASA Jet Propulsion Laboratory — Center for Near Earth Object Studies",
            "reliability_tier": 1,
            "content": (
                "NASA CNEOS IMPACT RISK ASSESSMENT\n"
                "Object: 2024 HX4 (Helios-4) | Report Date: February 29, 2024\n\n"
                "Based on 14 months of optical and radar observations, our orbital solution "
                "for Helios-4 yields a Palermo Scale value of -2.7, corresponding to a "
                "cumulative 100-year Earth impact probability of 0.003% (3 in 100,000).\n\n"
                "This probability is consistent with background impact risk for objects of "
                "this size class (~240m diameter). The object will be observed again during "
                "its 2031 close approach, at which time uncertainty is expected to narrow.\n"
                "Risk classification: GREEN (routine monitoring)."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-004",
        domain="scientific",
        true_claim="The OCEAN-2 research station recorded a 2.1°C rise in Arctic surface water temperature over the past 30 years at monitoring station ARC-7.",
        true_number="2.1",
        true_entity="OCEAN-2 research station",
        true_date="2024-03-15",
        primary_source={
            "title": "Arctic Ocean Temperature Trends: 30-Year Analysis (ARC-7)",
            "source": "NOAA Arctic Research Program",
            "reliability_tier": 1,
            "content": (
                "NOAA ARCTIC RESEARCH PROGRAM — TECHNICAL REPORT\n"
                "Publication Date: March 15, 2024 | Station: ARC-7 (81.2°N, 14.7°E)\n\n"
                "30-YEAR TEMPERATURE TREND ANALYSIS (1994–2024)\n"
                "Sea surface temperature (annual mean):\n"
                "  1994 baseline: -1.2°C\n"
                "  2024 measurement: +0.9°C\n"
                "  Net change: +2.1°C over 30-year period\n\n"
                "The 2.1°C warming at ARC-7 is consistent with broader Arctic "
                "amplification patterns documented across NOAA monitoring stations. "
                "The rate of warming (0.07°C/year) exceeds the global mean ocean "
                "surface warming rate by a factor of approximately 3.5."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-005",
        domain="scientific",
        true_claim="Praxis Labs announced that its quantum processor achieved 99.7% gate fidelity in a 50-qubit configuration, verified by an independent third party.",
        true_number="99.7",
        true_entity="Praxis Labs",
        true_date="2024-04-10",
        primary_source={
            "title": "Praxis Labs Quantum Processor Benchmarking — Independent Verification Report",
            "source": "National Institute of Standards and Technology (NIST)",
            "reliability_tier": 1,
            "content": (
                "NIST QUANTUM COMPUTING VERIFICATION REPORT\n"
                "Subject: Praxis Labs QPU-50 Performance Assessment\n"
                "Date: April 10, 2024 | Contract: NIST-2024-QC-017\n\n"
                "EXECUTIVE SUMMARY\n"
                "At the request of Praxis Labs, NIST conducted an independent benchmarking "
                "study of the QPU-50 quantum processor in a 50-qubit configuration.\n\n"
                "KEY FINDINGS\n"
                "Single-qubit gate fidelity: 99.91% (±0.03%)\n"
                "Two-qubit gate fidelity: 99.7% (±0.08%)\n"
                "Measurement fidelity: 99.4% (±0.05%)\n\n"
                "The 99.7% two-qubit gate fidelity figure is verified as accurate. "
                "This represents the highest independently-verified two-qubit fidelity "
                "reported in the literature to date."
            ),
        },
        secondary_sources=[],
    ),
]

# Fast lookup by fact_id
SEED_FACT_MAP: Dict[str, SeedFact] = {f.fact_id: f for f in SEED_FACTS}


# ── Document content templates for Spreader ───────────────────────────────

# These fill-in templates generate realistic-sounding content at each tier/spread-stage

TIER2_NEWS_TEMPLATES = [
    (
        "Breaking: {entity} {action_verb} amid {context}\n\n"
        "{claim_sentence} According to sources familiar with the matter, "
        "the decision follows months of internal deliberation. "
        "Industry analysts noted the move signals a broader shift in {domain} strategy. "
        "A spokesperson for {entity} declined to comment beyond the official statement."
    ),
    (
        "{entity} ANNOUNCES {headline_action}\n\n"
        "{claim_sentence} "
        "The announcement comes as {context}. Observers say this could have significant "
        "implications for {domain} stakeholders. The full impact is expected to become "
        "clear over the coming months."
    ),
]

TIER3_BLOG_TEMPLATES = [
    (
        "WOW: {entity} just {exaggerated_action}??\n\n"
        "So I was reading about this and {claim_sentence} "
        "Honestly not surprised given what's been happening lately. "
        "Share this before it gets taken down. This is huge."
    ),
    (
        "THREAD: What the media won't tell you about {entity}\n\n"
        "1/ {claim_sentence}\n"
        "2/ And that's just the beginning. Do your own research.\n"
        "3/ Like and repost if you think people need to see this."
    ),
]

NOISE_TEMPLATES = [
    (
        "{unrelated_entity} reported quarterly results in line with analyst expectations. "
        "Revenue of ${unrelated_number}M was up {unrelated_pct}% year-over-year. "
        "CEO remarked conditions remain favourable heading into the next quarter."
    ),
    (
        "The {unrelated_org} released its annual report covering activities across "
        "{unrelated_domain}. Key highlights include expanded partnerships and "
        "a {unrelated_pct}% increase in participation. The report is available on their website."
    ),
]


def get_random_fact(domain: Optional[str] = None, rng: Optional[random.Random] = None) -> SeedFact:
    """Return a random SeedFact, optionally filtered by domain."""
    r = rng or random.Random()
    pool = [f for f in SEED_FACTS if domain is None or f.domain == domain]
    return r.choice(pool)


def get_all_fact_ids() -> List[str]:
    return [f.fact_id for f in SEED_FACTS]
