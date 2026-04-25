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

    SeedFact(
        fact_id="SCI-011",
        domain="scientific",
        true_claim="A clinical trial found that increasing dosage beyond 50 mg did not improve efficacy but increased adverse events by 18%.",
        true_number="50",
        true_entity="MedCore Research Group",
        true_date="2024-03-12",
        primary_source={
            "title": "Dose Escalation Study Report",
            "source": "Clinical Pharmacology Journal",
            "reliability_tier": 1,
            "content": (
                "Efficacy plateaued at 50 mg dosage, while adverse events increased by 18% at higher doses."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-012",
        domain="scientific",
        true_claim="A study reported that air pollution exposure above 35 µg/m³ was associated with a 27% increase in respiratory hospitalizations.",
        true_number="35",
        true_entity="Global Health Institute",
        true_date="2024-02-10",
        primary_source={
            "title": "Air Pollution and Respiratory Risk Study",
            "source": "Environmental Health Perspectives",
            "reliability_tier": 1,
            "content": (
                "Exposure above 35 µg/m³ correlated with a 27% rise in respiratory hospital admissions."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-013",
        domain="scientific",
        true_claim="Researchers found that the new battery material retained 91% capacity after 1,000 charge cycles compared to 76% for standard lithium-ion cells.",
        true_number="91",
        true_entity="Electra Materials Lab",
        true_date="2024-03-03",
        primary_source={
            "title": "Battery Longevity Study",
            "source": "Journal of Energy Storage",
            "reliability_tier": 1,
            "content": (
                "New material retained 91% capacity after 1,000 cycles versus 76% for conventional cells."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-014",
        domain="scientific",
        true_claim="A neuroscience study showed that memory retention improved by 13% when learning sessions were spaced over 3 days instead of 1 day.",
        true_number="13",
        true_entity="Cognitive Science Institute",
        true_date="2024-01-18",
        primary_source={
            "title": "Spaced Learning Study",
            "source": "Neuroscience Letters",
            "reliability_tier": 1,
            "content": (
                "Spacing sessions across 3 days improved retention by 13% compared to single-day learning."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-015",
        domain="scientific",
        true_claim="A meta-analysis found that diets high in processed sugar increased the risk of Type 2 diabetes by 22%, controlling for BMI.",
        true_number="22",
        true_entity="Global Nutrition Consortium",
        true_date="2024-02-25",
        primary_source={
            "title": "Sugar Intake and Diabetes Risk Meta-analysis",
            "source": "The Lancet Public Health",
            "reliability_tier": 1,
            "content": (
                "High sugar intake was associated with a 22% increased diabetes risk after adjusting for BMI."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-016",
        domain="scientific",
        true_claim="An astronomy survey detected 17 new exoplanets within 50 light-years, including 3 in the habitable zone.",
        true_number="17",
        true_entity="DeepSky Observatory",
        true_date="2024-03-21",
        primary_source={
            "title": "Nearby Exoplanet Survey Results",
            "source": "Astrophysical Journal",
            "reliability_tier": 1,
            "content": (
                "17 exoplanets were identified within 50 light-years, 3 of which lie in habitable zones."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-017",
        domain="scientific",
        true_claim="A marine biology study reported coral bleaching decreased by 9% in protected reef zones compared to unprotected areas.",
        true_number="9",
        true_entity="Oceanic Research Alliance",
        true_date="2024-04-01",
        primary_source={
            "title": "Coral Reef Protection Impact Study",
            "source": "Marine Ecology Progress Series",
            "reliability_tier": 1,
            "content": (
                "Protected reefs showed 9% less bleaching compared to unprotected sites."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-018",
        domain="scientific",
        true_claim="A genetics study identified a mutation that increased disease risk by 3.5 times in affected individuals.",
        true_number="3.5",
        true_entity="Genomic Health Lab",
        true_date="2024-03-07",
        primary_source={
            "title": "Genetic Mutation Risk Analysis",
            "source": "Nature Genetics",
            "reliability_tier": 1,
            "content": (
                "Mutation carriers showed a 3.5x higher disease risk compared to non-carriers."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-019",
        domain="scientific",
        true_claim="A physics experiment measured a 0.02% deviation from predicted values under extreme conditions, within acceptable error margins.",
        true_number="0.02",
        true_entity="Quantum Dynamics Lab",
        true_date="2024-02-28",
        primary_source={
            "title": "High-Energy Physics Measurement Report",
            "source": "Physical Review Letters",
            "reliability_tier": 1,
            "content": (
                "Observed deviation was 0.02%, consistent with theoretical uncertainty bounds."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="SCI-020",
        domain="scientific",
        true_claim="A behavioral study found that participants exposed to blue light before sleep had 25% lower melatonin levels.",
        true_number="25",
        true_entity="Sleep and Circadian Lab",
        true_date="2024-01-29",
        primary_source={
            "title": "Blue Light and Sleep Hormone Study",
            "source": "Journal of Sleep Research",
            "reliability_tier": 1,
            "content": (
                "Melatonin levels were reduced by 25% in participants exposed to blue light before sleep."
            ),
        },
        secondary_sources=[],
    ),


    SeedFact(
        fact_id="FIN-001",
        domain="financial",
        true_claim="The central bank raised interest rates by 0.75 percentage points in Q2 2024, bringing the benchmark rate from 5.25% to 6.0% to curb inflation.",
        true_number="0.75",
        true_entity="National Central Bank",
        true_date="2024-05-18",
        primary_source={
            "title": "Monetary Policy Statement Q2 2024",
            "source": "National Central Bank",
            "reliability_tier": 1,
            "content": (
                "The policy rate was increased by 75 basis points from 5.25% to 6.0% "
                "in response to persistent inflation above target levels."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="FIN-002",
        domain="financial",
        true_claim="Country X reported GDP growth of 3.4% in 2023, while inflation averaged 6.1%, resulting in negative real income growth.",
        true_number="3.4",
        true_entity="Country X",
        true_date="2024-01-10",
        primary_source={
            "title": "National Economic Survey 2023",
            "source": "Ministry of Finance",
            "reliability_tier": 1,
            "content": (
                "GDP growth was 3.4% while inflation averaged 6.1%, eroding real income gains."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="FIN-003",
        domain="financial",
        true_claim="The unemployment rate declined from 7.2% to 5.8% over 18 months, while labor force participation increased by 1.6 percentage points.",
        true_number="5.8",
        true_entity="Labor Statistics Bureau",
        true_date="2024-03-30",
        primary_source={
            "title": "Labor Market Report 2024",
            "source": "Labor Statistics Bureau",
            "reliability_tier": 1,
            "content": (
                "Unemployment fell to 5.8% from 7.2%, alongside a 1.6 percentage point increase in participation."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="GEO-001",
        domain="geopolitical",
        true_claim="Country A and Country B signed a trade agreement reducing tariffs on 82% of goods over a 5-year phased schedule.",
        true_number="82",
        true_entity="Country A",
        true_date="2024-04-12",
        primary_source={
            "title": "Bilateral Trade Agreement Summary",
            "source": "Ministry of Trade",
            "reliability_tier": 1,
            "content": (
                "Tariffs will be reduced on 82% of traded goods over five years under the agreement."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="GEO-002",
        domain="geopolitical",
        true_claim="Defense spending increased by 11% in 2024, with 35% allocated to modernization programs and 20% to personnel costs.",
        true_number="11",
        true_entity="Defense Ministry",
        true_date="2024-02-14",
        primary_source={
            "title": "Defense Budget Allocation Report",
            "source": "Defense Ministry",
            "reliability_tier": 1,
            "content": (
                "Total defense spending rose by 11%, with allocations of 35% to modernization and 20% to personnel."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="TECH-001",
        domain="technology",
        true_claim="The new AI model reduced inference latency by 38% while increasing compute costs by 22% compared to the previous version.",
        true_number="38",
        true_entity="Synapse AI Labs",
        true_date="2024-03-25",
        primary_source={
            "title": "Model Performance Benchmark Report",
            "source": "Synapse AI Labs",
            "reliability_tier": 1,
            "content": (
                "Latency improved by 38%, while compute costs increased by 22% due to larger model size."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="TECH-002",
        domain="technology",
        true_claim="A distributed system upgrade increased throughput by 2.4x but introduced a 15% rise in tail latency under peak load.",
        true_number="2.4",
        true_entity="CloudScale Systems",
        true_date="2024-02-05",
        primary_source={
            "title": "System Scalability Evaluation",
            "source": "CloudScale Engineering",
            "reliability_tier": 1,
            "content": (
                "Throughput improved 2.4x, though tail latency increased by 15% during peak traffic."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="LAW-001",
        domain="legal",
        true_claim="The new regulation imposes fines of up to $2 million or 4% of annual revenue, whichever is higher, for non-compliance.",
        true_number="4",
        true_entity="Regulatory Authority",
        true_date="2024-01-01",
        primary_source={
            "title": "Regulatory Compliance Framework 2024",
            "source": "Regulatory Authority",
            "reliability_tier": 1,
            "content": (
                "Penalties include fines up to $2M or 4% of annual global revenue, whichever is greater."
            ),
        },
        secondary_sources=[],
    ),

    SeedFact(
        fact_id="LAW-002",
        domain="legal",
        true_claim="A court ruling determined that companies must disclose breaches within 72 hours if more than 10,000 users are affected.",
        true_number="72",
        true_entity="Federal Court",
        true_date="2024-03-19",
        primary_source={
            "title": "Cybersecurity Disclosure Ruling",
            "source": "Federal Court Records",
            "reliability_tier": 1,
            "content": (
                "Disclosure is required within 72 hours for breaches affecting over 10,000 users."
            ),
        },
        secondary_sources=[],
    ),



    SeedFact(
        fact_id="MUN-006",
        domain="municipal",
        true_claim="Harborview City Council voted to rezone 140 acres of former industrial land as mixed-use development, expected to add 2,400 housing units by 2027.",
        true_number="140",
        true_entity="Harborview City Council",
        true_date="2024-03-11",
        primary_source={
            "title": "Harborview City Council — Ordinance 2024-18: Eastport Industrial Rezoning",
            "source": "Harborview Office of the City Clerk",
            "reliability_tier": 1,
            "content": (
                "HARBORVIEW CITY COUNCIL — ORDINANCE 2024-18\n"
                "Adopted: March 11, 2024 | Public Hearing: February 26, 2024\n\n"
                "SUBJECT: Rezoning of Eastport Industrial Corridor to Mixed-Use District MX-3\n\n"
                "WHEREAS the Eastport Industrial Corridor (140 acres, bounded by Dock Street, "
                "Terminal Avenue, and the Harborview Shipping Canal) has remained underutilised "
                "since closure of the Consolidated Freight terminal in 2019;\n\n"
                "NOW THEREFORE the City Council hereby rezones said 140 acres from Industrial "
                "Heavy (IH) to Mixed-Use District MX-3, permitting residential, retail, and "
                "light commercial development.\n\n"
                "Development projections per Planning Department Impact Analysis (Feb 2024):\n"
                "  Residential units (projected): 2,400 (mix of market-rate and affordable)\n"
                "  Retail/commercial floor area: 180,000 sq ft\n"
                "  Target completion: Phase 1 by Q4 2025; full build-out by end of 2027\n\n"
                "Affordable housing requirement: 20% of units at or below 80% AMI.\n"
                "Environmental remediation cost-sharing: developer-funded ($12M bond required).\n"
                "VOTE: 6-1 in favour. Councillor Obata dissenting (traffic impact concerns)."
            ),
        },
        secondary_sources=[
            {
                "title": "Harborview Planning Department — Eastport Corridor Impact Analysis",
                "source": "Harborview Department of Planning and Zoning",
                "reliability_tier": 1,
                "content": (
                    "EASTPORT CORRIDOR MIXED-USE IMPACT ANALYSIS — FEBRUARY 2024\n\n"
                    "Site area: 140 acres | Proposed district: MX-3\n"
                    "Projected new housing units: 2,400 (1,920 market-rate, 480 affordable)\n"
                    "Projected population addition: ~4,200 residents\n"
                    "Traffic impact: +18% AM peak hour volume on Dock Street corridor.\n"
                    "Transit mitigation recommended: Bus Rapid Transit extension (cost: $34M, "
                    "subject to separate Council vote).\n"
                    "Environmental: Former petroleum storage — Remediation Order ER-2021-44 "
                    "requires soil and groundwater certification before building permits."
                ),
            },
        ],
    ),
 
    SeedFact(
        fact_id="MUN-007",
        domain="municipal",
        true_claim="Pinebrook Township passed a resolution requiring all new municipal buildings to meet LEED Gold certification standards, effective January 2025.",
        true_number="2025",
        true_entity="Pinebrook Township",
        true_date="2024-04-02",
        primary_source={
            "title": "Pinebrook Township — Resolution 2024-R-21: Green Building Standards",
            "source": "Pinebrook Township Clerk",
            "reliability_tier": 1,
            "content": (
                "PINEBROOK TOWNSHIP BOARD OF TRUSTEES\n"
                "Resolution 2024-R-21 | Adopted: April 2, 2024\n\n"
                "SUBJECT: Mandatory Green Building Standards for Municipal Construction\n\n"
                "RESOLVED: Effective January 1, 2025, all new Township-owned building "
                "construction projects with an estimated cost exceeding $500,000 must achieve "
                "a minimum LEED Gold certification rating as defined by the U.S. Green Building "
                "Council at the time of permit application.\n\n"
                "Scope: Applies to new construction only; renovation projects must target LEED "
                "Silver as a minimum.\n"
                "Cost impact estimate (Township Facilities Dept., March 2024): 3-6% premium "
                "over conventional construction, offset by projected 22% reduction in annual "
                "utility expenditure over 20-year lifecycle.\n"
                "Implementation: Director of Facilities shall update Township procurement "
                "specifications by November 1, 2024.\n"
                "VOTE: 5-0 unanimous."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="MUN-008",
        domain="municipal",
        true_claim="Clearwater County approved a $28.5M bond measure for water infrastructure upgrades, passing with 61% voter approval at the November 2023 election.",
        true_number="28.5",
        true_entity="Clearwater County",
        true_date="2023-11-07",
        primary_source={
            "title": "Clearwater County — Measure W Canvass of Returns, November 2023",
            "source": "Clearwater County Registrar of Voters",
            "reliability_tier": 1,
            "content": (
                "CLEARWATER COUNTY REGISTRAR OF VOTERS\n"
                "Official Canvass of Returns — General Election, November 7, 2023\n\n"
                "MEASURE W: WATER INFRASTRUCTURE GENERAL OBLIGATION BOND\n"
                "Ballot question: Shall Clearwater County issue general obligation bonds in the "
                "amount of $28,500,000 for the purposes of replacing aging water mains (est. "
                "average age: 67 years), upgrading the Eastside Water Treatment Plant filtration "
                "system, and installing SCADA monitoring across 14 pump stations?\n\n"
                "YES votes: 18,247 (61.3%)\n"
                "NO votes:  11,511 (38.7%)\n"
                "Total ballots cast: 29,758 | Registered voters: 54,112\n"
                "Turnout: 55.0%\n\n"
                "Required threshold for passage: 55% (water district bonds). MEASURE PASSES.\n"
                "Bond issuance authorised; first series expected Q2 2024.\n"
                "Estimated annual debt service: $1.6M over 20 years."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── CORPORATE ─────────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="CORP-006",
        domain="corporate",
        true_claim="Harmon Aerospace secured a $1.2B fixed-price contract from the U.S. Air Force for 24 next-generation surveillance drones, deliverable over 48 months.",
        true_number="1.2",
        true_entity="Harmon Aerospace",
        true_date="2024-01-18",
        primary_source={
            "title": "Harmon Aerospace Industries — SEC Form 8-K, January 2024",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "HARMON AEROSPACE INDUSTRIES, INC. — CURRENT REPORT (FORM 8-K)\n"
                "Date of Report: January 18, 2024\n\n"
                "ITEM 1.01 — ENTRY INTO MATERIAL DEFINITIVE AGREEMENT\n\n"
                "On January 15, 2024, Harmon Aerospace Industries, Inc. was awarded Contract "
                "FA8625-24-C-6601 by the United States Air Force Life Cycle Management Center.\n\n"
                "Contract type: Firm-Fixed-Price (FFP)\n"
                "Contract value: $1,200,000,000\n"
                "Scope: Design, manufacture, test, and delivery of 24 HA-9 High-Altitude Long-"
                "Endurance Surveillance Unmanned Aircraft Systems (HALE-SUAS), including "
                "associated ground control stations, logistics support packages, and operator "
                "training programme.\n"
                "Period of performance: 48 months from contract effective date.\n"
                "Deliverable schedule: 6 units per year beginning in month 18.\n\n"
                "This contract represents Harmon's largest single-award defence contract and "
                "is expected to contribute approximately $285M to annual revenue beginning FY2025. "
                "The Company's total backlog as of January 18, 2024 is approximately $4.8B."
            ),
        },
        secondary_sources=[
            {
                "title": "Harmon Aerospace — Q4 2023 Earnings Call Transcript (excerpt)",
                "source": "Harmon Aerospace Investor Relations",
                "reliability_tier": 2,
                "content": (
                    "CEO Remarks (January 18, 2024): We are extremely pleased to announce the "
                    "award of the HA-9 HALE-SUAS contract, valued at $1.2 billion. Delivery of "
                    "24 aircraft over 48 months will provide meaningful revenue visibility. "
                    "This win validates our investment in autonomous flight systems over the "
                    "past six years. We expect this programme to be margin-accretive by year two."
                ),
            },
        ],
    ),
 
    SeedFact(
        fact_id="CORP-007",
        domain="corporate",
        true_claim="NovaBridge Financial Group completed its acquisition of Sterling Community Bank for $340M, adding $2.1B in assets and 38 branch locations.",
        true_number="340",
        true_entity="NovaBridge Financial Group",
        true_date="2024-02-29",
        primary_source={
            "title": "NovaBridge Financial Group — SEC Form 8-K (Acquisition Completion)",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "NOVABRIDGE FINANCIAL GROUP, INC. — CURRENT REPORT (FORM 8-K)\n"
                "Date of Report: February 29, 2024\n\n"
                "ITEM 2.01 — COMPLETION OF ACQUISITION OR DISPOSITION OF ASSETS\n\n"
                "On February 29, 2024, NovaBridge Financial Group, Inc. completed its previously "
                "announced acquisition of Sterling Community Bank (OTCQB: STCB) pursuant to the "
                "Merger Agreement dated October 4, 2023.\n\n"
                "Consideration: $340,000,000 in cash ($18.72 per Sterling share at closing).\n"
                "Assets acquired: approximately $2.1 billion (as of December 31, 2023)\n"
                "Loan portfolio acquired: $1.4 billion\n"
                "Deposits assumed: $1.75 billion\n"
                "Branch locations: 38 (across 4 states: OH, KY, IN, WV)\n\n"
                "The acquisition expands NovaBridge's total asset base to approximately $9.8B "
                "and increases its branch count to 124. Core deposit premium paid: 6.2%. "
                "Regulatory approvals received from Federal Reserve and OCC on February 12, 2024."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="CORP-008",
        domain="corporate",
        true_claim="Solaris Energy reported that its utility-scale battery storage segment generated $412M in revenue in FY2023, representing 31% of total company revenue.",
        true_number="412",
        true_entity="Solaris Energy",
        true_date="2024-03-08",
        primary_source={
            "title": "Solaris Energy Corp — Annual Report (Form 10-K FY2023)",
            "source": "U.S. Securities and Exchange Commission EDGAR",
            "reliability_tier": 1,
            "content": (
                "SOLARIS ENERGY CORP — ANNUAL REPORT (FORM 10-K)\n"
                "Fiscal Year Ended: December 31, 2023 | Filed: March 8, 2024\n\n"
                "SEGMENT RESULTS — FISCAL YEAR 2023\n\n"
                "The Company reports three operating segments: Solar Generation, Battery Storage, "
                "and Energy Services.\n\n"
                "Battery Storage Segment:\n"
                "  Revenue FY2023:  $412,400,000\n"
                "  Revenue FY2022:  $238,100,000\n"
                "  Year-over-year growth: +73.2%\n"
                "  % of total company revenue: 31.0%\n"
                "  Adjusted EBITDA margin: 22.4%\n\n"
                "Battery Storage revenue growth was driven by three large grid-scale projects "
                "commissioned in California (180 MWh), Texas (240 MWh), and Arizona (160 MWh), "
                "and by expanded service contract revenue from prior-year installations.\n\n"
                "Total Company Revenue FY2023: $1,330,200,000 (+28.6% year-over-year)."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="CORP-009",
        domain="corporate",
        true_claim="PackageFast Logistics reduced last-mile delivery costs by 11.4% in 2023 after deploying autonomous delivery robots across seven metropolitan markets.",
        true_number="11.4",
        true_entity="PackageFast Logistics",
        true_date="2024-02-14",
        primary_source={
            "title": "PackageFast Logistics — FY2023 Annual Report and Operational Review",
            "source": "PackageFast Logistics Corp (privately held — investor report)",
            "reliability_tier": 1,
            "content": (
                "PACKAGEFAST LOGISTICS — FY2023 ANNUAL OPERATIONAL REVIEW\n"
                "Distributed to shareholders: February 14, 2024\n\n"
                "AUTONOMOUS DELIVERY PROGRAMME — 2023 RESULTS\n\n"
                "In Q1 2023, PackageFast began commercial deployment of its PF-R3 autonomous "
                "ground delivery robots across seven markets: Austin, Denver, Nashville, "
                "Columbus, Portland, Salt Lake City, and Richmond.\n\n"
                "Fleet deployed by December 31, 2023: 1,840 units\n"
                "Deliveries completed by autonomous units: 3.2 million (18% of total last-mile volume)\n"
                "Operational uptime average: 91.4%\n\n"
                "COST IMPACT:\n"
                "Last-mile cost per package (FY2022, human-only): $4.83\n"
                "Last-mile cost per package (FY2023, hybrid fleet): $4.28\n"
                "Reduction: $0.55 per package (-11.4%)\n\n"
                "Full rollout to 22 markets planned for 2024. Projected unit economics improve "
                "further at scale (target: -18% vs FY2022 baseline by end of 2025)."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── SCIENTIFIC ─────────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="SCI-021",
        domain="scientific",
        true_claim="A randomised controlled trial found that cognitive behavioural therapy reduced insomnia severity scores by 42% over 8 weeks, outperforming sleep medication at 6-month follow-up.",
        true_number="42",
        true_entity="National Sleep Research Consortium",
        true_date="2024-01-09",
        primary_source={
            "title": "CBT-I vs Pharmacotherapy in Chronic Insomnia: A Randomised Trial",
            "source": "JAMA Internal Medicine",
            "reliability_tier": 1,
            "content": (
                "RANDOMISED CONTROLLED TRIAL — JAMA Internal Medicine\n"
                "Published: January 9, 2024 | DOI: 10.1001/jamainternmed.2023.7841\n\n"
                "DESIGN: Multi-site RCT, n=312 adults with chronic insomnia disorder (DSM-5). "
                "Arms: (A) CBT for Insomnia (CBT-I, 6 sessions); (B) zolpidem 10 mg nightly; "
                "(C) combined; (D) placebo. Primary endpoint: Insomnia Severity Index (ISI) "
                "score reduction at 8 weeks.\n\n"
                "RESULTS (8 weeks):\n"
                "  CBT-I arm: Mean ISI reduction of 42% (from 18.4 to 10.7, p<0.001)\n"
                "  Zolpidem arm: Mean ISI reduction of 29% (from 18.1 to 12.8, p<0.001)\n"
                "  Combined arm: Mean ISI reduction of 44% (from 18.6 to 10.4)\n"
                "  Placebo: Mean ISI reduction of 9% (from 18.3 to 16.6)\n\n"
                "6-MONTH FOLLOW-UP:\n"
                "CBT-I arm maintained 38% ISI reduction from baseline. Zolpidem arm had "
                "partially regressed to 14% reduction (rebound insomnia observed in 31% of "
                "participants who discontinued medication).\n\n"
                "CONCLUSION: CBT-I produces durable insomnia reduction superior to "
                "pharmacotherapy at follow-up. Combination therapy offers no significant "
                "advantage over CBT-I alone at 6 months.\n\n"
                "National Sleep Research Consortium, 7-centre study."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="SCI-022",
        domain="scientific",
        true_claim="Researchers demonstrated that a novel mRNA vaccine candidate for influenza produced neutralising antibodies in 94% of participants, with reactogenicity comparable to seasonal flu shots.",
        true_number="94",
        true_entity="Vaxon Biosciences",
        true_date="2024-03-19",
        primary_source={
            "title": "Phase I/II Immunogenicity of mRNA-FLU-4v: An Open-Label Dose-Escalation Study",
            "source": "The Lancet Infectious Diseases",
            "reliability_tier": 1,
            "content": (
                "ORIGINAL ARTICLE — The Lancet Infectious Diseases\n"
                "Published: March 19, 2024 | DOI: 10.1016/S1473-3099(24)00081-X\n\n"
                "BACKGROUND: mRNA-FLU-4v is a quadrivalent mRNA influenza vaccine candidate "
                "encoding haemagglutinin antigens of two influenza A and two influenza B strains.\n\n"
                "METHODS: Open-label, dose-escalation Phase I/II trial, n=248 healthy adults "
                "aged 18-65. Three dosing cohorts (25 µg, 50 µg, 100 µg). Primary endpoints: "
                "seroconversion rate (≥4-fold rise in haemagglutination inhibition titre) and "
                "reactogenicity at Day 28.\n\n"
                "RESULTS:\n"
                "Seroconversion (neutralising antibody response) at 50 µg dose: 94.3% (n=234 "
                "evaluable; 95% CI: 90.4-97.2%).\n"
                "Geometric mean titre ratio vs. licensed quadrivalent inactivated vaccine (IIV4): "
                "1.8-fold higher across all four strains.\n\n"
                "Reactogenicity: Injection-site pain (Grade 1-2): 68%; fatigue: 41%; fever >38°C: "
                "9.3%. No Grade 4 or serious adverse events attributable to vaccine.\n"
                "Profile described as comparable to currently licensed seasonal influenza vaccines.\n\n"
                "CONCLUSION: mRNA-FLU-4v at 50 µg demonstrates robust immunogenicity with "
                "acceptable reactogenicity. Phase III efficacy trial initiation planned for Q3 2024.\n"
                "Sponsor: Vaxon Biosciences."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="SCI-023",
        domain="scientific",
        true_claim="A 10-year longitudinal cohort study found that adults who slept fewer than 6 hours per night had a 1.7-fold higher risk of developing hypertension compared to those sleeping 7-8 hours.",
        true_number="1.7",
        true_entity="Framingham Sleep Cohort Study",
        true_date="2024-02-06",
        primary_source={
            "title": "Sleep Duration and Incident Hypertension: 10-Year Follow-Up of the Framingham Sleep Cohort",
            "source": "Circulation",
            "reliability_tier": 1,
            "content": (
                "ORIGINAL RESEARCH — Circulation\n"
                "Published: February 6, 2024 | DOI: 10.1161/CIRCULATIONAHA.123.066714\n\n"
                "STUDY DESIGN: Prospective longitudinal cohort. n=4,810 adults aged 30-65, "
                "free of hypertension at enrolment (2010-2013). Follow-up duration: 10 years "
                "(through 2022-2023). Sleep duration assessed by wrist actigraphy at baseline "
                "and at 5-year follow-up.\n\n"
                "PRIMARY OUTCOME: Incident hypertension (SBP ≥130 or DBP ≥80 mmHg, or "
                "antihypertensive medication initiation).\n\n"
                "RESULTS:\n"
                "Short sleep (<6 hours/night): hazard ratio (HR) for incident hypertension = "
                "1.73 (95% CI: 1.48-2.02), adjusted for age, sex, BMI, smoking, alcohol, "
                "physical activity, and baseline blood pressure.\n"
                "Reference group: 7-8 hours/night.\n"
                "Long sleep (>9 hours/night): HR = 1.29 (95% CI: 1.04-1.60).\n"
                "10-year cumulative incidence of hypertension: 38.4% in short sleepers vs "
                "22.6% in reference group.\n\n"
                "CONCLUSION: Short sleep duration is independently associated with a 1.7-fold "
                "elevated risk of incident hypertension. Clinical screening for sleep duration "
                "should be integrated into cardiovascular risk assessment.\n"
                "Framingham Sleep Cohort Study, Boston University School of Medicine."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="SCI-024",
        domain="scientific",
        true_claim="A field study documented that kelp forest canopy cover declined by 74% along 400 km of the Northern California coast between 2008 and 2023, driven primarily by sea urchin population explosions.",
        true_number="74",
        true_entity="Pacific Marine Ecology Lab",
        true_date="2024-03-28",
        primary_source={
            "title": "Catastrophic Decline of Northern California Kelp Forests: 2008-2023 Assessment",
            "source": "Science of the Total Environment",
            "reliability_tier": 1,
            "content": (
                "RESEARCH ARTICLE — Science of the Total Environment\n"
                "Published: March 28, 2024 | DOI: 10.1016/j.scitotenv.2024.171843\n\n"
                "ABSTRACT\n"
                "We quantified change in giant kelp (Macrocystis pyrifera) and bull kelp "
                "(Nereocystis luetkeana) canopy cover along 400 km of Northern California "
                "coastline (Mendocino to Sonoma counties) using Landsat satellite imagery "
                "(2008-2023), aerial surveys (2018, 2021, 2023), and in situ transect data.\n\n"
                "FINDINGS:\n"
                "Kelp canopy area (2008): 10,490 hectares\n"
                "Kelp canopy area (2023):  2,727 hectares\n"
                "Decline: -74.0% over 15 years\n\n"
                "Primary driver: Purple sea urchin (Strongylocentrotus purpuratus) barrens now "
                "cover 58% of formerly forested rocky reef habitat. Urchin density increased "
                "from a mean of 0.8/m² (2008) to 11.3/m² (2023), attributable to the collapse "
                "of the sunflower sea star (Pycnopodia helianthoides) due to sea star wasting "
                "disease beginning in 2013-2014.\n"
                "Secondary driver: Marine heat waves (2014-2016, 2019-2020) elevated SST by "
                "+2.1°C above climatological average during critical sporophyte recruitment windows.\n\n"
                "Restoration trials using urchin removal showed partial recovery (24% canopy "
                "return within 18 months at treated sites).\n"
                "Pacific Marine Ecology Lab, UC Davis Bodega Marine Laboratory."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="SCI-025",
        domain="scientific",
        true_claim="A materials science study reported a solid-state electrolyte achieving ionic conductivity of 32 mS/cm at room temperature, surpassing liquid electrolyte benchmarks for the first time.",
        true_number="32",
        true_entity="Westbrook Institute for Advanced Materials",
        true_date="2024-04-03",
        primary_source={
            "title": "Room-Temperature Superionic Conductivity in Argyrodite-Class Solid Electrolytes",
            "source": "Nature Materials",
            "reliability_tier": 1,
            "content": (
                "ARTICLE — Nature Materials\n"
                "Published: April 3, 2024 | DOI: 10.1038/s41563-024-01842-6\n\n"
                "We report a lithium argyrodite-class solid electrolyte (Li6.6PS5.4Cl0.6, "
                "designated WI-AE9) exhibiting room-temperature ionic conductivity of 32 mS/cm, "
                "as measured by electrochemical impedance spectroscopy across three independent "
                "sample batches.\n\n"
                "This exceeds the ionic conductivity of conventional liquid carbonate electrolytes "
                "(typically 10-15 mS/cm) by approximately 2-3 fold, marking the first solid "
                "electrolyte to surpass the liquid benchmark under ambient conditions without "
                "applied pressure.\n\n"
                "Key structural features enabling superionic transport:\n"
                "  - Optimised Cl⁻/S²⁻ ratio (0.6) expanding Li-Li site hopping pathways\n"
                "  - Nano-domain ordering confirmed via synchrotron XRD and cryo-STEM\n"
                "  - Electrochemical stability window: 0.0-5.2 V vs. Li/Li⁺\n\n"
                "Pellet-cell cycling (Li-metal | WI-AE9 | LFP cathode) demonstrated stable "
                "capacity retention of 96.8% after 500 cycles at 1C rate.\n\n"
                "Westbrook Institute for Advanced Materials, in collaboration with Argonne National Laboratory."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── FINANCIAL ─────────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="FIN-004",
        domain="financial",
        true_claim="The national pension fund reported a 7.3% annual return for FY2023, outperforming its 6.5% benchmark, while managing $482B in total assets.",
        true_number="7.3",
        true_entity="National Pension Investment Board",
        true_date="2024-03-15",
        primary_source={
            "title": "National Pension Investment Board — FY2023 Annual Investment Report",
            "source": "National Pension Investment Board",
            "reliability_tier": 1,
            "content": (
                "NATIONAL PENSION INVESTMENT BOARD\n"
                "Annual Investment Performance Report — Fiscal Year 2023\n"
                "Published: March 15, 2024\n\n"
                "FUND PERFORMANCE SUMMARY\n"
                "Total assets under management (December 31, 2023): $482.3 billion\n"
                "Net investment return FY2023: 7.3%\n"
                "Policy benchmark return FY2023: 6.5%\n"
                "Active return (alpha): +0.8 percentage points\n"
                "5-year annualised return: 8.1% (benchmark: 7.6%)\n"
                "10-year annualised return: 9.4% (benchmark: 8.9%)\n\n"
                "ASSET ALLOCATION (December 31, 2023):\n"
                "  Global equities: 42% ($202.6B)\n"
                "  Fixed income:    28% ($135.0B)\n"
                "  Private equity:  14% ($67.5B)\n"
                "  Real assets:     10% ($48.2B)\n"
                "  Hedge funds:      6% ($29.0B)\n\n"
                "The FY2023 outperformance was primarily driven by the global equities portfolio "
                "(+10.1% vs. MSCI All Country World Index +8.6%) and private equity (+12.4%), "
                "partially offset by fixed income underperformance during the rate hike cycle.\n"
                "Funded status: 96.2% (actuarial valuation basis)."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="FIN-005",
        domain="financial",
        true_claim="The Meridian Stock Exchange launched a carbon credit futures contract in January 2024, with $1.4B in notional value traded in its first 60 days.",
        true_number="1.4",
        true_entity="Meridian Stock Exchange",
        true_date="2024-03-12",
        primary_source={
            "title": "Meridian Stock Exchange — Carbon Futures Launch Report (60-Day Review)",
            "source": "Meridian Stock Exchange Market Operations",
            "reliability_tier": 1,
            "content": (
                "MERIDIAN STOCK EXCHANGE — CARBON FUTURES CONTRACT\n"
                "60-Day Market Performance Review | Published: March 12, 2024\n\n"
                "CONTRACT LAUNCH: January 8, 2024\n"
                "Contract: MSE Carbon Credit Futures (ticker: MCCF)\n"
                "Underlying: Verified Carbon Units (VCU) per VERRA VCS standard\n"
                "Contract size: 1,000 VCUs (= 1,000 metric tons CO2e)\n"
                "Settlement: Monthly, cash-settled at VERRA spot price\n\n"
                "60-DAY TRADING STATISTICS (January 8 – March 8, 2024):\n"
                "Total contracts traded: 1,418,204\n"
                "Total notional value: $1.42 billion\n"
                "Average daily volume: 24,451 contracts/day\n"
                "Peak daily volume: 41,200 contracts (February 14, following EU ETS announcement)\n"
                "Open interest (March 8): 87,340 contracts\n\n"
                "Participant breakdown: Asset managers 38%, Compliance buyers 29%, "
                "Proprietary traders 21%, Corporate hedgers 12%.\n"
                "Exchange fee revenue from MCCF: $2.8M in first 60 days.\n"
                "MSE Chief Market Officer: 'Exceeded our 90-day volume projections by 40%.'"
            ),
        },
        secondary_sources=[],
    ),
 
    # ── GEOPOLITICAL ──────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="GEO-003",
        domain="geopolitical",
        true_claim="The Kestrel Alliance summit concluded with a joint declaration committing 14 member states to contribute a minimum of 2.1% of GDP to collective defence spending by 2028.",
        true_number="2.1",
        true_entity="Kestrel Alliance",
        true_date="2024-02-22",
        primary_source={
            "title": "Kestrel Alliance — Valletta Summit Joint Declaration, February 2024",
            "source": "Kestrel Alliance Secretariat",
            "reliability_tier": 1,
            "content": (
                "KESTREL ALLIANCE — VALLETTA SUMMIT JOINT DECLARATION\n"
                "Valletta, Malta | February 22, 2024\n\n"
                "We, the Heads of State and Government of the 14 Member States of the Kestrel "
                "Alliance, assembled at the Valletta Summit, hereby declare:\n\n"
                "ARTICLE 3 — DEFENCE INVESTMENT PLEDGE\n"
                "Each Member State commits to achieving a minimum defence expenditure of 2.1% "
                "of Gross Domestic Product by December 31, 2028, to be measured using NATO-"
                "compatible accounting standards.\n\n"
                "Progress reporting: Annual submission to the Kestrel Alliance Defence "
                "Investment Monitoring Body (DIMB) beginning 2025.\n"
                "Enforcement: Member States falling below 1.8% GDP in any fiscal year will "
                "be subject to review by the Alliance Council.\n\n"
                "Current status (2023 data): 6 of 14 member states meet the 2.1% threshold. "
                "Aggregate Alliance defence spending (2023): $284B. Projected aggregate at full "
                "compliance: $341B.\n\n"
                "Signed by all 14 member state representatives. Declaration enters into force "
                "immediately upon signature."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="GEO-004",
        domain="geopolitical",
        true_claim="Country Y imposed export controls on six categories of advanced semiconductor manufacturing equipment, restricting sales to entities in designated countries without government licence.",
        true_number="6",
        true_entity="Country Y Ministry of Commerce",
        true_date="2024-01-31",
        primary_source={
            "title": "Country Y — Export Administration Regulation Amendment Notice 2024-EC-003",
            "source": "Country Y Ministry of Commerce — Export Control Bureau",
            "reliability_tier": 1,
            "content": (
                "COUNTRY Y MINISTRY OF COMMERCE\n"
                "EXPORT ADMINISTRATION REGULATION — AMENDMENT NOTICE 2024-EC-003\n"
                "Effective Date: January 31, 2024\n\n"
                "SUBJECT: Addition of Advanced Semiconductor Manufacturing Equipment to "
                "Export Control List (ECL) — Categories A through F\n\n"
                "The Ministry of Commerce hereby amends the Export Administration Regulations "
                "to add six categories of advanced semiconductor manufacturing equipment to the "
                "Export Control List, requiring mandatory export licences for sales, transfers, "
                "or re-exports to entities in Designated Countries (as listed in Annex II).\n\n"
                "CONTROLLED CATEGORIES:\n"
                "  Category A: Extreme ultraviolet (EUV) lithography systems\n"
                "  Category B: Deep ultraviolet (DUV) immersion lithography systems (≤193nm)\n"
                "  Category C: Atomic layer deposition (ALD) equipment for gate dielectric\n"
                "  Category D: High-aspect-ratio etch systems (>50:1 selectivity)\n"
                "  Category E: Metrology systems for sub-3nm process nodes\n"
                "  Category F: Wafer bonding equipment for advanced 3D packaging\n\n"
                "Licence applications reviewed within 45 business days. Exports pending licence "
                "approval are suspended pending determination.\n"
                "Violations: subject to fines up to ¥50M or 5x transaction value, and/or "
                "criminal prosecution under the Export Control Law (2020 revision)."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── TECHNOLOGY ────────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="TECH-003",
        domain="technology",
        true_claim="Axon Semiconductor taped out a 2nm test chip using gate-all-around nanosheet transistors, achieving a transistor density of 300 million transistors per mm².",
        true_number="300",
        true_entity="Axon Semiconductor",
        true_date="2024-04-08",
        primary_source={
            "title": "Axon Semiconductor — Technical White Paper: 2nm GAA Nanosheet Process Node",
            "source": "Axon Semiconductor Corp",
            "reliability_tier": 1,
            "content": (
                "AXON SEMICONDUCTOR CORP — TECHNICAL WHITE PAPER\n"
                "2nm Gate-All-Around Nanosheet Process Node Development\n"
                "Published: April 8, 2024\n\n"
                "EXECUTIVE SUMMARY\n"
                "Axon Semiconductor announces successful tape-out of the AX-2N test vehicle, "
                "implementing our second-generation gate-all-around (GAA) nanosheet transistor "
                "architecture on a 300mm wafer using EUV lithography (5-layer EUV patterning).\n\n"
                "KEY PROCESS METRICS:\n"
                "  Transistor density:        300 million transistors / mm²\n"
                "  Nanosheet width:           5-7 nm (3-stack configuration)\n"
                "  Gate pitch:                42 nm\n"
                "  Metal pitch:               20 nm (M0 layer)\n"
                "  Drive current improvement: +18% vs. our 3nm FinFET node at Vdd=0.75V\n"
                "  Leakage current reduction: -35% vs. 3nm node\n"
                "  Static power reduction:    -28% at equivalent performance point\n\n"
                "VERIFICATION: Independent yield and density measurements conducted by "
                "Fraunhofer Institute for Integrated Systems (IISB) under NDA. Transistor "
                "density of 300M/mm² confirmed via e-beam inspection and SEM cross-section.\n\n"
                "Volume production readiness: targeted for Q2 2025. Three foundry customers "
                "have signed advance capacity reservation agreements (details confidential)."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="TECH-004",
        domain="technology",
        true_claim="A large-scale study of 3.2 million open-source repositories found that 68% contained at least one dependency with a known critical or high-severity CVE as of January 2024.",
        true_number="68",
        true_entity="OpenSource Security Alliance",
        true_date="2024-02-27",
        primary_source={
            "title": "State of Open Source Security 2024: Dependency Vulnerability Prevalence Study",
            "source": "OpenSource Security Alliance",
            "reliability_tier": 1,
            "content": (
                "OPENSOURCE SECURITY ALLIANCE\n"
                "State of Open Source Security Report 2024\n"
                "Published: February 27, 2024\n\n"
                "STUDY OVERVIEW\n"
                "The OpenSource Security Alliance analysed 3,200,000 public repositories hosted "
                "on GitHub, GitLab, and Bitbucket, sampled as of January 15, 2024. Dependency "
                "manifests (package.json, requirements.txt, pom.xml, go.mod, Cargo.toml, "
                "Gemfile) were parsed and cross-referenced against the National Vulnerability "
                "Database (NVD) and GitHub Advisory Database.\n\n"
                "KEY FINDING: 68.4% of repositories (2,188,800) contained at least one direct "
                "or transitive dependency with a CVE rated Critical (CVSS ≥9.0) or High "
                "(CVSS 7.0-8.9) that had not been patched in the repository's dependency lock file.\n\n"
                "Breakdown by ecosystem:\n"
                "  npm (JavaScript):    72.1% of repos with critical/high CVE\n"
                "  PyPI (Python):       65.8%\n"
                "  Maven (Java):        61.4%\n"
                "  RubyGems:            74.3%\n"
                "  Go modules:          44.7% (lowest exposure)\n\n"
                "Median time from CVE publication to repository patch: 47 days.\n"
                "17.2% of repositories had at least one dependency with a CVE published >1 year "
                "prior that remained unpatched.\n"
                "Methodology: automated scanning only; manual triage not performed."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── LEGAL ─────────────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="LAW-003",
        domain="legal",
        true_claim="The Federal Appeals Court ruled that non-compete agreements exceeding 18 months are presumptively unenforceable for employees earning below $95,000 annually.",
        true_number="18",
        true_entity="Federal Appeals Court, Third Circuit",
        true_date="2024-03-05",
        primary_source={
            "title": "Delmar Technologies v. Rossi — Opinion, Third Circuit Court of Appeals",
            "source": "United States Court of Appeals, Third Circuit",
            "reliability_tier": 1,
            "content": (
                "UNITED STATES COURT OF APPEALS FOR THE THIRD CIRCUIT\n"
                "Case No. 23-1847 | Decided: March 5, 2024\n\n"
                "DELMAR TECHNOLOGIES, INC. v. ROSSI et al.\n\n"
                "OPINION BY CHIEF JUDGE HARTMANN:\n\n"
                "This appeal concerns the enforceability of non-compete agreements under the "
                "common law rule of reason as applied by courts in this Circuit.\n\n"
                "We hold today that non-compete covenants exceeding eighteen (18) months in "
                "duration are presumptively unenforceable when applied to employees whose "
                "total annual compensation at the time of separation is below $95,000. This "
                "presumption may be rebutted only upon a showing by clear and convincing evidence "
                "that the employee had access to highly specialised trade secrets that would cause "
                "substantial irreparable harm if disclosed within the extended restriction period.\n\n"
                "Delmar's 24-month non-compete clause applied to respondent Rossi, a software "
                "developer earning $82,000 annually, fails this test. The district court's "
                "injunction is VACATED and the matter REMANDED with instructions to enter "
                "judgment for the respondent.\n\n"
                "SCOPE: Today's holding applies to all employment non-compete agreements "
                "governed by the laws of states within this Circuit. It does not address "
                "non-solicitation agreements, confidentiality agreements, or sale-of-business covenants.\n"
                "PANEL: Chief Judge Hartmann, Circuit Judges Okonkwo and De Santis."
            ),
        },
        secondary_sources=[],
    ),
 
    SeedFact(
        fact_id="LAW-004",
        domain="legal",
        true_claim="A class action settlement required AdMetrics Corp to pay $215M to approximately 4.8 million consumers for unlawful collection of biometric data without informed consent.",
        true_number="215",
        true_entity="AdMetrics Corp",
        true_date="2024-01-24",
        primary_source={
            "title": "Yuen et al. v. AdMetrics Corp — Final Settlement Approval Order",
            "source": "U.S. District Court, Northern District of Illinois",
            "reliability_tier": 1,
            "content": (
                "UNITED STATES DISTRICT COURT\n"
                "NORTHERN DISTRICT OF ILLINOIS — EASTERN DIVISION\n"
                "Case No. 1:21-cv-04881 | Order Entered: January 24, 2024\n\n"
                "YUEN et al. v. ADMETRICS CORP\n"
                "FINAL ORDER AND JUDGMENT APPROVING CLASS ACTION SETTLEMENT\n\n"
                "The Court has considered the parties' Joint Motion for Final Approval of Class "
                "Settlement and finds the settlement fair, reasonable, and adequate pursuant to "
                "Fed. R. Civ. P. 23(e).\n\n"
                "SETTLEMENT TERMS APPROVED:\n"
                "Total settlement fund: $215,000,000 (two hundred fifteen million dollars)\n"
                "Class definition: All Illinois residents from whom AdMetrics Corp collected, "
                "captured, or stored voiceprint or facial geometry data via its FocusMetric® "
                "advertising effectiveness platform between January 1, 2017 and June 30, 2022 "
                "without first obtaining informed written consent, in violation of the Illinois "
                "Biometric Information Privacy Act (BIPA), 740 ILCS 14/1 et seq.\n"
                "Class size: approximately 4,800,000 members\n"
                "Per-claimant payment: approximately $44.79 (estimated, subject to claim rate)\n"
                "Attorneys' fees: 25% of fund ($53.75M), subject to lodestar cross-check\n"
                "Cy pres recipient: Electronic Frontier Foundation ($2.15M)\n\n"
                "Claim submission deadline: April 30, 2024. AdMetrics admits no liability."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── EDUCATIONAL ───────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="EDU-001",
        domain="educational",
        true_claim="The Grantham Unified School District implemented a 1-to-1 device programme for all K-8 students in 2023, reporting a 19% improvement in standardised reading assessment scores over 12 months.",
        true_number="19",
        true_entity="Grantham Unified School District",
        true_date="2024-01-30",
        primary_source={
            "title": "Grantham USD — Technology Integration Impact Report: Year 1 (2023-24)",
            "source": "Grantham Unified School District, Office of Research and Assessment",
            "reliability_tier": 1,
            "content": (
                "GRANTHAM UNIFIED SCHOOL DISTRICT\n"
                "Technology Integration Impact Report — Year 1 (SY 2022-23 to SY 2023-24)\n"
                "Published: January 30, 2024\n\n"
                "PROGRAMME OVERVIEW\n"
                "In August 2022, the District launched a 1-to-1 Chromebook initiative for all "
                "students in grades K-8 (11,240 students across 17 schools), funded by a "
                "combination of ESSER III funds ($4.1M) and local bond proceeds.\n\n"
                "READING ASSESSMENT RESULTS (YEAR 1)\n"
                "Assessment instrument: NWEA MAP Reading Growth\n"
                "Baseline (Fall 2022 administration): District mean RIT score: 204.7\n"
                "Year-1 endpoint (Fall 2023 administration): District mean RIT score: 243.6\n"
                "Improvement: +38.9 RIT points; represents +19.0% improvement relative to baseline.\n\n"
                "Comparison: State average improvement over same period: +9.4%.\n"
                "Effect size (Cohen's d): 0.61 — moderate to large.\n\n"
                "Disaggregated data:\n"
                "  English Learners: +22.3% (largest subgroup gain)\n"
                "  Students with IEPs: +14.7%\n"
                "  Socioeconomically disadvantaged: +20.1%\n\n"
                "Confounding factors acknowledged: Simultaneous literacy coaching programme "
                "implemented in same period; causal attribution to device programme alone is "
                "not established. Year 2 study with matched control group underway.\n"
                "Report prepared by: Director of Research and Assessment, January 2024."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── ENVIRONMENTAL ─────────────────────────────────────────────────────
 
    SeedFact(
        fact_id="ENV-001",
        domain="environmental",
        true_claim="The Upper Dawnton River Basin restoration project removed 94,000 metric tons of legacy phosphorus sediment between 2020 and 2023, reducing harmful algal bloom frequency by 67%.",
        true_number="94000",
        true_entity="Upper Dawnton River Basin Authority",
        true_date="2024-02-12",
        primary_source={
            "title": "Upper Dawnton River Basin Restoration Programme — Phase II Completion Report",
            "source": "Upper Dawnton River Basin Authority / EPA Region 5",
            "reliability_tier": 1,
            "content": (
                "UPPER DAWNTON RIVER BASIN AUTHORITY\n"
                "Phase II Restoration Programme — Completion and Outcome Report\n"
                "Co-published with EPA Region 5 | February 12, 2024\n\n"
                "PROGRAMME SUMMARY\n"
                "The Phase II Sediment Remediation Programme (2020-2023) targeted legacy "
                "phosphorus-laden sediment deposited during historical agricultural and industrial "
                "discharge across the Upper Dawnton River Basin (1,840 km² catchment area).\n\n"
                "REMEDIATION OUTCOMES\n"
                "Total phosphorus-laden sediment removed: 94,000 metric tons\n"
                "Removal method: Hydraulic dredging (63%) and mechanical capping (37%)\n"
                "River reaches treated: 148 km of main stem and major tributaries\n"
                "Cost: $127M (federal: $89M, state: $28M, local: $10M)\n\n"
                "ECOLOGICAL RESPONSE (2020-2023 monitoring data)\n"
                "Harmful algal bloom (HAB) events per season:\n"
                "  Baseline (2017-2019 average): 18 events/season\n"
                "  Post-remediation (2023): 6 events/season\n"
                "  Reduction: -12 events (-66.7%, rounded to 67%)\n\n"
                "Total phosphorus concentration (main stem, August mean):\n"
                "  Baseline: 0.18 mg/L\n"
                "  2023: 0.09 mg/L (-50%)\n\n"
                "Dissolved oxygen compliance (>5 mg/L threshold): 91% of monitoring days in "
                "2023 vs. 58% baseline. Macroinvertebrate index scores improved from 'Poor' "
                "to 'Fair' in 11 of 14 monitored reaches."
            ),
        },
        secondary_sources=[],
    ),
 
    # ── HEALTHCARE / PUBLIC HEALTH ─────────────────────────────────────────
 
    SeedFact(
        fact_id="HLT-001",
        domain="healthcare",
        true_claim="A national hospital readmission reduction programme cut 30-day all-cause readmission rates from 15.8% to 11.2% over three years, saving an estimated $890M in Medicare expenditure.",
        true_number="11.2",
        true_entity="Centre for Hospital Quality Improvement",
        true_date="2024-03-01",
        primary_source={
            "title": "National Readmission Reduction Initiative — 3-Year Outcome Report",
            "source": "Centre for Hospital Quality Improvement / Department of Health and Human Services",
            "reliability_tier": 1,
            "content": (
                "CENTRE FOR HOSPITAL QUALITY IMPROVEMENT\n"
                "National Readmission Reduction Initiative (NRRI) — 3-Year Outcome Report\n"
                "Published: March 1, 2024\n\n"
                "PROGRAMME BACKGROUND\n"
                "The NRRI enrolled 4,218 acute care hospitals beginning January 2021, implementing "
                "structured care transition protocols including: pharmacist-led medication "
                "reconciliation, nurse navigator post-discharge calls (Days 2, 7, and 14), "
                "and primary care appointment scheduling within 7 days of discharge.\n\n"
                "PRIMARY OUTCOME — 30-DAY ALL-CAUSE READMISSION RATE\n"
                "Baseline (calendar year 2020): 15.8%\n"
                "Year 1 (2021):                 14.4%\n"
                "Year 2 (2022):                 12.6%\n"
                "Year 3 (2023):                 11.2%\n"
                "Total reduction:               -4.6 percentage points (-29.1%)\n\n"
                "FINANCIAL IMPACT (Medicare Fee-for-Service claims analysis)\n"
                "Averted readmissions (2021-2023): approximately 218,000 hospitalisation episodes\n"
                "Estimated gross Medicare savings: $890M over 3 years\n"
                "Programme implementation cost (federal grant): $124M\n"
                "Net savings: $766M\n\n"
                "CONDITION-SPECIFIC REDUCTIONS:\n"
                "  Heart failure: 15.8% → 9.7% (-38.6%)\n"
                "  Pneumonia: 14.1% → 10.4% (-26.2%)\n"
                "  COPD: 20.2% → 15.8% (-21.8%)\n\n"
                "Participating hospitals with ≥80% protocol adherence achieved 2× the "
                "readmission reduction compared to low-adherence sites."
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
