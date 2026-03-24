"""Ontology / ingestion path: LLM structured extraction from raw text."""

EXTRACTION_PROMPT = """Extract structured information from this resume/job description text.
Return ONLY valid JSON — no markdown fences, no preamble:
{
  "technical_skills": ["list of specific technical skills, tools, languages, frameworks"],
  "soft_skills": ["list of soft skills"],
  "certifications": ["list of certifications like AWS SAA, CISSP, PMP"],
  "experience_years": <integer of total years, or 0>,
  "education": ["list of degrees"],
  "domain": "primary industry domain",
  "seniority": "junior|mid|senior|lead|executive"
}
Map synonyms to canonical forms (e.g., "data ingestion workflows" → "ETL pipelines").
TEXT:
"""

ONTOLOGY_EXTRACTION_SYSTEM = "Extract structured data. Return ONLY valid JSON."
